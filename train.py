# pylint:disable=missing-class-docstring,missing-function-docstring,missing-module-docstring,g-long-lambda,line-too-long

import os

import google3
import tensorflow as tf
import tensorflow_hub as hub


class Config(models.BaseModelConfig):
  exp_name = 'retinopathy'
  img_size = 224
  train_batch_size = 8
  val_batch_size = 8
  eval_every = 100
  eval_iters = 10
  architecture = 'basic'
  finetune = False


def parse_ds(example, img_size):
  features = tf.io.parse_single_example(
      example, {
          'image': tf.io.FixedLenFeature((), tf.string, b''),
          'content': tf.io.FixedLenFeature((), tf.string, b''),
          'level': tf.io.FixedLenFeature((), tf.int64, -1),
      })
  img = tf.reshape(features['content'], ())
  if img == tf.constant(b''):
    img = tf.zeros((img_size, img_size, 3))
    level = tf.constant(-1, dtype=tf.int64)
  else:
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize([img], (img_size, img_size))[0]
  features['content'] = img
  return features['image'], features['content'], features['level']


def is_train(image_name):
  return tf.strings.to_hash_bucket_fast(image_name, 10) != 0


class Model(tf.keras.Model):

  def __init__(self, config: Config):
    super().__init__()
    self.cnn1 = tf.keras.layers.Conv2D(
        128, (3, 3), padding='same', activation='relu')  # (128, 128, 128)
    self.cnn2 = tf.keras.layers.Conv2D(
        128, (3, 3), strides=(2, 2), padding='same',
        activation='relu')  # (64, 64, 128)
    self.cnn3 = tf.keras.layers.Conv2D(
        128, (3, 3), strides=(2, 2), padding='same',
        activation='relu')  # (32, 32, 128)
    self.cnn4 = tf.keras.layers.Conv2D(
        128, (3, 3), strides=(2, 2), padding='same',
        activation='relu')  # (16, 16, 128)
    self.cnn5 = tf.keras.layers.Conv2D(
        128, (3, 3), strides=(2, 2), padding='same',
        activation='relu')  # (8, 8, 128)
    self.cnn6 = tf.keras.layers.Conv2D(
        128, (3, 3), strides=(2, 2), padding='same',
        activation='relu')  # (4, 4, 128)
    self.flatten = tf.keras.layers.Flatten()
    self.out = tf.keras.layers.Dense(5)

  def call(self, inp, training=False):
    img = inp
    img = self.cnn1(img)
    img = self.cnn2(img)
    img = self.cnn3(img)
    img = self.cnn4(img)
    img = self.cnn5(img)
    img = self.cnn6(img)
    return self.out(self.flatten(img))


class Trainer(models.BaseTrainer):

  def build(self, _):
    ds = (
        tf.data.TFRecordDataset(
            gfile.Glob(
                os.path.join(
                    constants.DS_PATH, 'all_tfrecord/all.tfrecord-*'))).map(
                        lambda x: parse_ds(x, self.config.img_size)).filter(
                            lambda _, _2, level: level != -1).repeat())
    self.ds_train = ds.filter(lambda image, _, _2: is_train(image)).batch(
        self.config.train_batch_size).prefetch(2)
    self.ds_val = ds.filter(lambda image, _, _2: not is_train(image)).batch(
        self.config.val_batch_size).prefetch(2)
    if self.config.architecture == 'basic':
      self.model = Model(self.config)
    elif self.config.architecture == 'resnet50':
      self.model = tf.keras.Sequential([
          hub.KerasLayer(
              'https://tfhub.dev/tensorflow/resnet_50/feature_vector/1',
              trainable=self.config.finetune),
          tf.keras.layers.Dense(5),
      ])
    elif self.config.architecture == 'resnet50_empty':
      input_shape = (self.config.img_size, self.config.img_size, 3)
      inputs = tf.keras.Input(shape=input_shape)
      outputs = tf.keras.applications.ResNet50(
          weights=None, input_shape=input_shape, classes=5)(
              inputs)
      self.model = tf.keras.Model(inputs, outputs)
    else:
      assert False

    self.optimizer = tf.keras.optimizers.Adam(0.001)
    self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    self.accuracy = tf.keras.metrics.Accuracy()
    self.checkpointer = tf.train.Checkpoint(
        model=self.model,
        optimizer=self.optimizer,
        total_steps=self.total_steps)

  def train_step(self, batch):
    _, content, level = batch
    with tf.GradientTape() as tape:
      pred_logit = self.model(content, training=True)
      loss = self.loss_fn(level, pred_logit)
      grads = tape.gradient(loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
      self.hist.scalar('loss', loss, self.total_steps)
      self.hist.scalar(
          'accuracy/train',
          tf.reduce_mean(
              tf.cast(tf.math.argmax(pred_logit, axis=-1) == level,
                      tf.float32)), self.total_steps)

  def eval_start(self):
    self.accuracy.reset_states()

  def eval_step(self, batch):
    _, content, level = batch
    pred_logit = self.model(content, training=True)
    self.accuracy.update_state(
        tf.math.argmax(pred_logit, axis=-1)[..., tf.newaxis], level[...,
                                                                    tf.newaxis])

  def eval_end(self):
    self.hist.scalar('accuracy/eval', self.accuracy.result(), self.total_steps)
