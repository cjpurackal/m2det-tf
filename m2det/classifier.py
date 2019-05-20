import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.activations import softmax


def simple_classifier(config, feature):
	num_classes = 2 + 1  # background
	num_anchors = config["anchors"]["num_anchors"]
	cls = Conv2D(
		filters=num_classes*num_anchors,
		kernel_size=(3, 3), strides=(1, 1),
		padding='same', use_bias=True
		)(feature)
	cls = BatchNormalization()(cls)
	sh = feature.shape
	cls = softmax(
		tf.reshape(
			cls,
			[-1, sh[1] * sh[2] * num_anchors, num_classes]
			)
		)
	return cls
