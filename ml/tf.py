"""Tensorflow utils."""

import tensorflow as tf


def bytes_f(val):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))


def str_f(val):
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[val.encode('utf-8')]))


def int64_f(val):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))


def float_f(val):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[val]))


def bytes_list_f(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def strs_f(values):
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(
          value=[val.encode('utf-8') for val in values]))


def int64s_f(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def floats_f(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

