import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.regularizers import l2
from m2det.utils import bilinear_upsampler


class VGG16():

	def __init__(self, inputs=None, config=None):	
		self.inputs = inputs
		self.config = config

	def forward(self, return_tail=False):

		weight_decay = 0.000

		#conv1_1
		x = Conv2D(64, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(self.inputs)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)

		#conv1_1
		x = Conv2D(64, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)

		# pool1
		x = MaxPooling2D(pool_size=(2, 2))(x)
		
		# conv2_1
		x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)
		
		# conv2_1
		x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		
		# pool2
		x = MaxPooling2D(pool_size=(2, 2))(x)
		
		# conv3_1
		x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)
		
		# conv3_2
		x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)
		
		# conv3_2
		x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)

		# pool3
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# conv4_1
		x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)

		# conv4_2
		x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)

		# conv4_3
		x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = conv4_3 = BatchNormalization()(x)

		# pool4
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# conv5_1
		x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)

		# conv5_2
		x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)

		# conv5_3
		x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		conv5_3 = BatchNormalization()(x)

		if return_tail:
			#for experiments
			return conv5_3
		else:
			feat1 = bilinear_upsampler(conv4_3, (self.config["model"]["backbone_feature1_size"]))
			feat2 = bilinear_upsampler(conv5_3, (self.config["model"]["backbone_feature2_size"]))
			#x = MaxPooling2D(pool_size=(2, 2))(x)
			#x = Dropout(0.5)(x)
			return feat1, feat2