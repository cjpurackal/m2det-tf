import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.activations import softmax
from m2det.utils import flatten


def predictor(config, features):
	num_classes = 2 + 1 # background class
	num_anchors = config["anchors"]["num_anchors"]
	all_cls = []
	all_box = []
	for feature in features:
		cls = Conv2D(
			filters=num_anchors*num_classes,
			kernel_size=(3, 3), strides=(1, 1),
			padding='same',use_bias=True
			)(feature)
		cls = BatchNormalization()(cls)
		all_cls.append(flatten(cls))
		reg = Conv2D(
			filters=num_anchors*4,kernel_size=(3, 3),
			strides=(1, 1),padding='same',
			use_bias=True
			)(feature)
		reg = BatchNormalization()(reg)
		all_box.append(flatten(reg))
	all_cls = tf.concat(all_cls, axis=1)
	all_box = tf.concat(all_box, axis=1)
	num_boxes = int(all_box.shape[-1]/4)
	all_cls = tf.reshape(all_cls, [-1, num_boxes, num_classes])
	all_cls = softmax(all_cls)
	all_box = tf.reshape(all_box, [-1, num_boxes, 4])
	return all_cls, all_box