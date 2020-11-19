import os
import enum

import numpy as np
import six
import tensorflow as tf

from absl import app
from ml import models
import big_gans
import constants


class Config(models.BaseModelConfig):
    exp_name = 'retinopathy'
    img_size = 224
    architecture = 'basic'
    finetune = False
    is_dev_eval_async = False
    clear_buffer_when_logging_plots = False
    # start:colab_only
    train_batch_size = 8
    val_batch_size = 8
    eval_every = 20
    eval_iters = 2
    # end:colab_only
    # start:uncomment_for_py
    # train_batch_size = 16
    # val_batch_size = 16
    # eval_every = 100
    # eval_iters = 10
    # dev_eval_iters = 2
    # eval_per_dev_eval = 2
    # end:uncomment_for_py
    log_plots_every = eval_iters


def parse_ds(example, img_size):
    features = tf.io.parse_single_example(example, {
        'image': tf.io.FixedLenFeature((), tf.string, b''),
        'content': tf.io.FixedLenFeature((), tf.string, b''),
        'level': tf.io.FixedLenFeature((), tf.int64, -10),
    })
    img = tf.reshape(features['content'], ())
    if img == tf.constant(b''):
        img = tf.zeros((img_size, img_size, 3))
        level = tf.constant(-1, dtype=tf.int64)
    else:
        img = tf.io.decode_jpeg(img)
        img = tf.image.resize([img], (img_size, img_size))[0]
    img = img / 128 - 1
    features['content'] = img
    return features['image'], features['content'], features['level']


def is_train(image_name, level):
    return tf.logical_or(
        tf.strings.to_hash_bucket_fast(image_name, 10) != 0,
        level == -1)


class LayerWithBatchNorm(tf.keras.layers.Layer):

    def __init__(self, layer_class, **kwargs):
        super(LayerWithBatchNorm, self).__init__()
        self._layer_class = layer_class
        self._activation = kwargs.get('activation', None)
        kwargs['activation'] = None
        self._kwargs = kwargs

    def build(self, input_shape):
        self._layer = self._layer_class(**self._kwargs)
        self._norm = tf.keras.layers.BatchNormalization()
        if isinstance(self._activation, six.string_types):
            self._activation = tf.keras.layers.Activation(self._activation)

    def call(self, inp):
        o = self._norm(self._layer(inp))
        if self._activation:
            o = self._activation(o)
        return o


class LayerConfig(enum.Enum):
    DEFAULT = 'default'
    BATCH_NORM_LEAKY_RELU = 'bn_lr'

    @staticmethod
    def Get(config):
        if not isinstance(config, six.string_types):
            return config
        return LayerConfig(config)


class BaseLayer(tf.keras.layers.Layer):
    def _Conv2DT(self,
                 filters,
                 kernel_size,
                 activation=None,
                 config='bn_lr',
                 reg=0.02,
                 **kwargs):
        config = LayerConfig.Get(config)
        if config == LayerConfig.BATCH_NORM_LEAKY_RELU:
            return LayerWithBatchNorm(
                tf.keras.layers.Conv2DTranspose,
                filters=filters,
                kernel_size=kernel_size,
                use_bias=False,
                activation=activation or tf.keras.layers.LeakyReLU(),
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(reg),
                bias_regularizer=tf.keras.regularizers.l2(reg),
                **kwargs)
        return tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            **kwargs)

    def _Conv2D(self,
                filters,
                kernel_size,
                activation=None,
                config='bn_lr',
                reg=0.02,
                **kwargs):
        config = LayerConfig.Get(config)
        if config == LayerConfig.BATCH_NORM_LEAKY_RELU:
            return LayerWithBatchNorm(
                tf.keras.layers.Conv2D,
                filters=filters,
                kernel_size=kernel_size,
                activation=activation or tf.keras.layers.LeakyReLU(),
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(reg),
                bias_regularizer=tf.keras.regularizers.l2(reg),
                **kwargs)
        return tf.keras.layers.Conv2D(filter=filters, kernel_size=kernel_size)

    def _Dense(self, dim, activation=None, config='bn_lr', reg=0.02, **kwargs):
        config = LayerConfig.Get(config)
        if config == LayerConfig.BATCH_NORM_LEAKY_RELU:
            return LayerWithBatchNorm(
                tf.keras.layers.Dense,
                units=dim,
                activation=activation or tf.keras.layers.LeakyReLU(),
                kernel_regularizer=tf.keras.regularizers.l2(reg),
                bias_regularizer=tf.keras.regularizers.l2(reg),
                **kwargs)
        return tf.keras.layers.Dense(dim, activation, **kwargs)


class ImageGenerator(BaseLayer):
    def __init__(self, reg=0.02, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(rate=0.4)
        self.dense1 = tf.keras.layers.Dense(
            7 * 7 * 64,
            activation=tf.keras.layers.ReLU(),
            kernel_regularizer=tf.keras.regularizers.l2(reg),
            bias_regularizer=tf.keras.regularizers.l2(reg))
        self.reshape1 = tf.keras.layers.Reshape((7, 7, 64))
        self.conv1t = self._Conv2DT(64, kernel_size=(5, 5), strides=(2, 2))
        self.conv2t = self._Conv2DT(64, kernel_size=(5, 5), strides=(2, 2))
        self.conv3t = self._Conv2DT(64, kernel_size=(5, 5), strides=(2, 2))
        self.conv4t = self._Conv2DT(64, kernel_size=(5, 5), strides=(2, 2))
        self.conv5t = self._Conv2DT(3, kernel_size=(5, 5), strides=(2, 2), activation='tanh')

    def call(self, o):
        o = self.dropout(self.dense1(o))
        o = self.reshape1(o)
        o = self.dropout(self.conv1t(o))
        o = self.dropout(self.conv2t(o))
        o = self.dropout(self.conv3t(o))
        o = self.dropout(self.conv4t(o))
        o = self.conv5t(o)
        assert o.shape[1:] == (224, 224, 3), o.shape
        return o


class ImageDiscriminator1(BaseLayer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.3)

        self.conv1 = self._Conv2D(64, kernel_size=(5, 5), strides=(2, 2))
        self.conv2 = self._Conv2D(64, kernel_size=(5, 5), strides=(2, 2))
        self.conv3 = self._Conv2D(64, kernel_size=(5, 5), strides=(2, 2))
        self.conv4 = self._Conv2D(64, kernel_size=(5, 5), strides=(2, 2))
        self.conv5 = self._Conv2D(64, kernel_size=(5, 5), strides=(2, 2))
        self.real_logit = tf.keras.layers.Dense(1)
        self.class_logit = tf.keras.layers.Dense(5)

    def call(self, o):
        # assert o.shape[1:] == (self.image_size, self.image_size), o.shape
        o = tf.keras.backend.expand_dims(o)
        o = self.dropout(self.conv1(o))
        o = self.dropout(self.conv2(o))
        o = self.dropout(self.conv3(o))
        o = self.dropout(self.conv4(o))
        o = self.dropout(self.conv5(o))
        o = self.flatten(o)
        return (
            self.real_logit(o),
            self.class_logit(o))


class ImageDiscriminator(BaseLayer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.3)

        self.resnet50 = tf.keras.applications.ResNet50()
        self.real_logit = tf.keras.layers.Dense(1)
        self.class_logit = tf.keras.layers.Dense(5)

    def call(self, o):
        # assert o.shape[1:] == (self.image_size, self.image_size), o.shape
        o = tf.keras.backend.expand_dims(o)
        o = self.resnet50(o, training=True)
        o = self.flatten(o)
        return (
            self.real_logit(o),
            self.class_logit(o))


def create_ds(ds_paths, img_size):
    np.random.shuffle(ds_paths)
    ds = tf.data.Dataset.from_tensor_slices(ds_paths)
    ds = (
        ds.interleave(lambda x: tf.data.TFRecordDataset([x]), cycle_length=16, block_length=1)
            .map(lambda x: parse_ds(x, img_size))
            .filter(lambda _, _2, level: level != -10)
            .repeat())
    return ds


class Trainer(models.BaseTrainer):
    def build(self, _):
        labelled_paths = gfile.Glob(os.path.join(constants.DS_PATH, 'all_tfrecord/all.tfrecord-*'))
        unlabelled_paths = gfile.Glob(os.path.join(constants.DS_PATH, 'test_tfrecord/test.tfrecord-*'))

        self.ds_train = (
            create_ds(labelled_paths + unlabelled_paths, self.config.img_size)
                .filter(lambda image, _, level: is_train(image, level))
                .batch(len(self.devices) * self.config.train_batch_size)
                .prefetch(2))
        self.ds_val = (
            create_ds(labelled_paths, self.config.img_size)
                .filter(lambda image, _, level: not is_train(image, level))
                .batch(len(self.devices) * self.config.val_batch_size)
                .prefetch(2))

        with self.mirrored_strategy.scope():
            # self.gen_model = ImageGenerator()
            self.gen_model = big_gans.BigGan()
            self.dis_model = ImageDiscriminator()
            self.gen_optimizer = tf.keras.optimizers.Adam(0.001)
            self.dis_optimizer = tf.keras.optimizers.Adam(0.001)

        # start:uncomment_for_py
        self.checkpointer = tf.train.Checkpoint(
            gen_model=self.gen_model,
            dis_model=self.dis_model,
            gen_optimizer=self.gen_optimizer,
            dis_optimizer=self.dis_optimizer,
            total_steps=self.total_steps)
        # end:uncomment_for_py

        self.scce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.real_accuracy = tf.keras.metrics.Accuracy()
        self.fake_accuracy = tf.keras.metrics.Accuracy()
        self.real_accuracy_ema = tf.Variable(0.0)
        self.fake_accuracy_ema = tf.Variable(0.0)
        self.dis_run_prob = tf.Variable(1.0)
        self.gen_run_prob = tf.Variable(1.0)
        self.class_accuracy = tf.keras.metrics.Accuracy()

    # start:uncomment_for_py
    @tf.function
    # end:uncomment_for_py
    def train_step(self, batch):
        _, content, level = batch
        batch_size = self.config.train_batch_size
        global_batch_size = len(self.devices) * batch_size
        zeros = tf.zeros((batch_size, 1))
        ones = tf.ones((batch_size, 1))
        dis_grad_factor = tf.cast(tf.random.uniform((), 0, 1) < self.dis_run_prob, tf.float32)
        gen_grad_factor = tf.cast(tf.random.uniform((), 0, 1) < self.gen_run_prob, tf.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            noise = tf.random.normal((batch_size, 7 * 128))
            gen_images = self.gen_model(noise, training=True)
            fake_dis_output = self.dis_model(gen_images, training=True)
            fake_loss_dis = tf.nn.compute_average_loss(
                self.bce(zeros, fake_dis_output[0]),
                global_batch_size=global_batch_size)
            fake_loss_gen = tf.nn.compute_average_loss(
                self.bce(ones, fake_dis_output[0]),
                global_batch_size=global_batch_size)

        # print('gen loop')
        gen_grads = tape1.gradient(fake_loss_gen, self.gen_model.trainable_variables)
        gen_grads = [g * gen_grad_factor if g is not None else None for g in gen_grads]
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.gen_model.trainable_variables))

        # print('dis fake loop')
        dis_grads = tape2.gradient(fake_loss_dis, self.dis_model.trainable_variables)
        dis_grads = [g * dis_grad_factor if g is not None else None for g in dis_grads]
        self.dis_optimizer.apply_gradients(zip(dis_grads, self.dis_model.trainable_variables))

        with tf.GradientTape() as tape:
            real_dis_output = self.dis_model(content, training=True)
            real_dis_loss_real = tf.nn.compute_average_loss(
                self.bce(ones, real_dis_output[0]),
                global_batch_size=global_batch_size)
            mask = tf.cast(level != -1, tf.float32)
            real_dis_loss_class = tf.nn.compute_average_loss(
                self.scce(
                    tf.cast(mask, tf.int64) * level,
                    real_dis_output[1], sample_weight=mask),
                global_batch_size=global_batch_size)

            # print('dis real loop')
            # print('mask', mask)
            # print('real_dis_loss_class', real_dis_loss_class)
            real_dis_loss = real_dis_loss_real + real_dis_loss_class
        dis_grads = tape.gradient(real_dis_loss, self.dis_model.trainable_variables)
        dis_grads = [g * dis_grad_factor if g is not None else None for g in dis_grads]
        self.dis_optimizer.apply_gradients(zip(dis_grads, self.dis_model.trainable_variables))

        return fake_loss_dis, fake_loss_gen, real_dis_loss_real, real_dis_loss_class, real_dis_loss

    def train_step_end(self, step, args):
        mean = lambda x: self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=None)
        fake_loss_dis, fake_loss_gen, real_dis_loss_real, real_dis_loss_class, real_dis_loss = args
        self.hist.scalar('dis/loss_fake', mean(fake_loss_dis), self.total_steps)
        self.hist.scalar('gen/loss_fake', mean(fake_loss_gen), self.total_steps)
        self.hist.scalar('dis/loss_real', mean(real_dis_loss_real), self.total_steps)
        self.hist.scalar('dis/loss_class', mean(real_dis_loss_class), self.total_steps)
        self.hist.scalar('dis/loss', mean(real_dis_loss), self.total_steps)

    def eval_start(self):
        self.real_accuracy.reset_states()
        self.class_accuracy.reset_states()

    # start:uncomment_for_py
    @tf.function
    # end:uncomment_for_py
    def eval_step(self, batch):
        _, content, level = batch
        zeros = tf.zeros((self.config.val_batch_size, 1))
        ones = tf.ones((self.config.val_batch_size, 1))
        real_acc, pred_logit = self.dis_model(content, training=False)
        self.class_accuracy.update_state(
            level[..., tf.newaxis],
            tf.math.argmax(pred_logit, axis=-1)[..., tf.newaxis])
        self.real_accuracy.update_state(
            ones, tf.cast(real_acc > 0, tf.int32))
        gen_images = self.gen_model(
            tf.random.normal((self.config.val_batch_size, 128 * 7)),
            training=False)
        fake_acc, _ = self.dis_model(gen_images, training=False)
        self.fake_accuracy.update_state(
            zeros, tf.cast(fake_acc > 0, tf.int32))

    def eval_end(self):
        self.real_accuracy_ema.assign(self.real_accuracy_ema * 0.8 + self.real_accuracy.result() * 0.2)
        self.fake_accuracy_ema.assign(self.fake_accuracy_ema * 0.8 + self.fake_accuracy.result() * 0.2)
        dis_accuracy_min = tf.math.minimum(self.real_accuracy_ema, self.fake_accuracy_ema)
        self.dis_run_prob.assign(tf.cond(dis_accuracy_min > 0.5, lambda: self.dis_run_prob * 0.8,
                                         lambda: self.dis_run_prob + (1 - self.dis_run_prob) * 0.2))
        self.gen_run_prob.assign(
            tf.cond(dis_accuracy_min > 0.5, lambda: self.gen_run_prob + (1 - self.gen_run_prob) * 0.2,
                    lambda: self.gen_run_prob * 0.8))
        self.hist.scalar('accuracy/eval_real', self.real_accuracy.result(), self.total_steps)
        self.hist.scalar('accuracy/eval_fake', self.fake_accuracy.result(), self.total_steps)
        self.hist.scalar('accuracy/eval_class', self.class_accuracy.result(), self.total_steps)
        self.hist.scalar('num/iters_dis', self.dis_run_prob.numpy(), self.total_steps)
        self.hist.scalar('num/iters_gen', self.gen_run_prob.numpy(), self.total_steps)

    def dev_eval_async(self, step=0):
        gen_images = self.gen_model(
            tf.random.normal((self.config.val_batch_size, 128 * 7)),
            training=False)
        self.hist.image('gen', gen_images, self.total_steps)


def main(_):
    trainer = Trainer(Config)
    trainer.start(None)


if __name__ == '__main__':
    app.run(main)
