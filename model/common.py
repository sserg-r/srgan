import numpy as np
import tensorflow as tf

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


def subpixel_conv2d(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


