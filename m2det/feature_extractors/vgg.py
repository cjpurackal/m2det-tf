import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from m2det.utils import bilinear_upsampler


class VGG16():

	def __init__(self, inputs, config):	
		self.inputs = inputs
		self.config = config

	def forward(self):

		weight_decay = 0.000

		#conv1_1
		x = layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(self.inputs)
		x = layers.Activation('relu')(x)
		x = layers.BatchNormalization()(x)

		#conv1_1
		x = layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		x = layers.BatchNormalization()(x)

		# pool1
		x = layers.MaxPooling2D(pool_size=(2, 2))(x)
		
		# conv2_1
		x = layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.4)(x)
		
		# conv2_1
		x = layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		x = layers.BatchNormalization()(x)
		
		# pool2
		x = layers.MaxPooling2D(pool_size=(2, 2))(x)
		
		# conv3_1
		x = layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.4)(x)
		
		# conv3_2
		x = layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.4)(x)
		
		# conv3_2
		x = layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		x = layers.BatchNormalization()(x)

		# pool3
		x = layers.MaxPooling2D(pool_size=(2, 2))(x)

		# conv4_1
		x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.4)(x)

		# conv4_2
		x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.4)(x)

		# conv4_3
		x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		x = conv4_3 = layers.BatchNormalization()(x)

		# pool4
		x = layers.MaxPooling2D(pool_size=(2, 2))(x)

		# conv5_1
		x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.4)(x)

		# conv5_2
		x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.4)(x)

		# conv5_3
		x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = layers.Activation('relu')(x)
		conv5_3 = layers.BatchNormalization()(x)

		feat1 = bilinear_upsampler(conv4_3, (self.config["model"]["backbone_feature1_size"]))
		feat2 = bilinear_upsampler(conv5_3, (self.config["model"]["backbone_feature2_size"]))
		#x = layers.MaxPooling2D(pool_size=(2, 2))(x)
		#x = layers.Dropout(0.5)(x)

		return feat1, feat2