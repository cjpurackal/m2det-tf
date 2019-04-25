import tensorflow as tf


def bilinear_upsampler(tensor, new_shape):
	return tf.image.resize_images(tensor, new_shape)
