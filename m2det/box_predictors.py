import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization


def simple_predictor(config, feature):
	num_anchors = config["anchors"]["num_anchors"]
	reg = Conv2D(
		filters=num_anchors*4, kernel_size=(3, 3),
		strides=(1, 1), padding='same',
		use_bias=True
		)(feature)
	reg = BatchNormalization()(reg)
	sh = reg.shape
	box = tf.reshape(reg, [-1, sh[1] * sh[2] * sh[3]])
	# shape (batch_size), (feature[1]* feature[2] * num_anchors * 4)
	return box
