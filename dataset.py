# pylint:disable=missing-class-docstring,missing-function-docstring,missing-module-docstring,g-long-lambda,line-too-long

import tqdm

import pandas as pd


ds = pd.read_csv(files.cache_locally(os.path.join(constants.DS_PATH, 'trainLabels.csv')))
ds['patient_id'] = ds['image'].apply(lambda x: x.split('_')[0])
ds['path'] = ds['image'].apply(lambda x: os.path.join(constants.DS_PATH, 'train', f'{x}.jpeg'))
ds = ds[ds['path'].apply(lambda p: gfile.Exists(p))]
tf_record_files = [
    tf.io.TFRecordWriter(os.path.join(DS_PATH, 'all_tfrecord', f'all.tfrecord-{i:05d}-00100'))
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