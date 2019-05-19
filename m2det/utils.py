import tensorflow as tf


def bilinear_upsampler(tensor, new_shape):
	return tf.image.resize(tensor, new_shape)