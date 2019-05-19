import tensorflow as tf


def bilinear_upsampler(tensor, new_shape):
	return tf.image.resize(tensor, new_shape)


def flatten(x):
    sh = x.shape
    return tf.reshape(x, [-1, sh[1] * sh[2] * sh[3]])