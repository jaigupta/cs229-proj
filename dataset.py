import os

import pandas as pd
import tensorflow as tf
import tqdm

from ml import tf as mltf
from ml import files
import constants


def read_bytes(path):
  with tf.io.gfile.GFile(path, 'rb') as f:
    return f.read()

def create_from_train_ds():
  ds = pd.read_csv(files.cache_locally(os.path.join(constants.DS_PATH, 'trainLabels.csv')))
  ds['patient_id'] = ds['image'].apply(lambda x: x.split('_')[0])
  ds['path'] = ds['image'].apply(lambda x: os.path.join(constants.DS_PATH, 'train', f'{x}.jpeg'))
  ds = ds[ds['path'].apply(lambda p: tf.io.gfile.exists(p))]
  tf_record_files = [
      tf.io.TFRecordWriter(os.path.join(constants.DS_PATH, 'all_tfrecord', f'all.tfrecord-{i:05d}-of-00100'))
      for i in range(100)]
  for image, path, level in tqdm.tqdm(ds[['image', 'path', 'level']].values):
    bucket = tf.strings.to_hash_bucket_fast(image, 100)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': mltf.str_f(image),
        'content': mltf.bytes_f(read_bytes(path)),
        'level': mltf.int64_f(level)
    }))
    tf_record_files[bucket].write(example.SerializeToString())

  for f in tf_record_files:
    f.close()

# create_from_train_ds()


def create_from_test_ds():
  test_ds_path = os.path.join(constants.DS_PATH, 'test')
  ds = pd.DataFrame()
  ds['image'] = [f.split('.')[0] for f in gfile.ListDir(test_ds_path)]
  ds['path'] = ds['image'].apply(lambda x: os.path.join(constants.DS_PATH, 'test', f'{x}.jpeg'))
  tf_record_files = [
      tf.io.TFRecordWriter(os.path.join(DS_PATH, 'test_tfrecord', f'test.tfrecord-{i:05d}-of-00100'))
      for i in range(100)]
  for image, path in tqdm.tqdm(ds[['image', 'path']].values):
    bucket = tf.strings.to_hash_bucket_fast(image, 100)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': mltf.str_f(image),
        'content': mltf.bytes_f(read_bytes(path)),
        'level': mltf.int64_f(-1)
    }))
    tf_record_files[bucket].write(example.SerializeToString())

  for f in tf_record_files:
    f.close()

# create_from_test_ds()
