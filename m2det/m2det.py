import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from m2det.feature_extractors import *
from m2det import FFM, TUM, SFAM
from m2det.box_predictors import simple_predictor
from m2det.classifier import simple_classifier


supported_backbones = {"vgg16":VGG16, "darknet21":Darknet21}

class M2det:
	def __init__(self, config):
		self.config = config
		assert config["model"]["backbone"] in supported_backbones.keys(), "%s Usupported backbone!"%(backbone)
		self.backbone = supported_backbones[config["model"]["backbone"]]

	def forward(self, imgs):
		f1, f2 = self.backbone(imgs, self.config).forward()		
		ffm = FFM(f1, f2)

		#collecting decoder outputs from TUMs
		decoder_outs = []
		for i in range(self.config["model"]["tums"]):
			if i == 0:
				features = ffm.v1()
				features = Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=256)(features)
			else:
				features = ffm.v2(decoder_outs[i-1][-1])
			decoder_outs.append(TUM(self.config,features).forward())
		
		#constructing mlfpn using SFAM
		mlfpn = SFAM(self.config, decoder_outs).forward()
		boxes = []
		classes = []
		for feature_cube in mlfpn:
			box_pred = simple_predictor(self.config, feature_cube)
			cls_pred = simple_classifier(self.config, feature_cube)
			boxes.append(box_pred)
			classes.append(cls_pred)

		all_box = tf.concat(boxes, axis=1)
		all_classes = tf.concat(classes, axis=1)
		all_box = tf.reshape(all_box, [-1, int(all_box.shape[1]/4), 4])
		y_pred = tf.concat([all_box,all_classes], axis=2)

		return y_pred
