import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.activations import relu


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.activations import relu


class TUM:
	def __init__(self, config, features):
		self.config = config
		self.features = features

	def bilinear_upsampler(self, tensor, shape):
		return tf.image.resize_images(tensor, shape)

	def forward(self):
		#   encoder
		encoder_outs = []
		for i in range(self.config["model"]["scales"]):
			if i == 0:
				conv_out = Conv2D(
						kernel_size=(3, 3), strides=(2, 2),
						filters=256, padding='same')(self.features)
			elif i == self.config["model"]["scales"] - 1:
				conv_out = Conv2D(
						kernel_size=(3, 3), strides=(2, 2),
						filters=256)(encoder_outs[i-1])
			else:
				conv_out = Conv2D(
						kernel_size=(3, 3), strides=(2, 2),
						filters=256, padding='same')(encoder_outs[i-1])
			encoder_outs.append(relu(BatchNormalization()(conv_out)))

		#   decoder
		decoder_outs = []
		bs_outs = []
		for i in range(self.config["model"]["scales"]+1):
			if i == 0:
				dec_out = Conv2D(
						kernel_size=(1, 1), strides=(1, 1),
						filters=128)(encoder_outs[-1])
				decoder_outs.append(relu(BatchNormalization()(dec_out)))
				bs_outs.append(self.bilinear_upsampler(
					encoder_outs[-1], encoder_outs[-2].shape[1:3]))
			else:
				conv_out = Conv2D(
						kernel_size=(3, 3), strides=(1, 1),
						filters=256)(bs_outs[i-1])
				conv_out = relu(BatchNormalization()(conv_out))
				if i != (self.config["model"]["scales"]):
					bs_out = encoder_outs[-i-1] +
					self.bilinear_upsampler(
						conv_out, encoder_outs[-i-1].shape[1:3])
				else:
					bs_out = self.features +
					self.bilinear_upsampler(
						conv_out, self.features.shape[1:3])
				dec_out = Conv2D(
					kernel_size=(1, 1), strides=(1, 1),
					filters=128)(bs_out)
				decoder_outs.append(relu(BatchNormalization()(dec_out)))
				bs_outs.append(bs_out)
		return decoder_outs
		"""
			conv0_out = Conv2D(
				kernel_size=(3, 3), strides=(2, 2),
				filters=256, padding='same')(self.features)
			conv0_out = relu(BatchNormalization()(conv0_out))
			conv1_out = Conv2D(
				kernel_size=(3, 3), strides=(2, 2),
				filters=256, padding='same')(conv0_out)
			conv1_out = relu(BatchNormalization()(conv1_out))
			conv2_out = Conv2D(
				kernel_size=(3, 3), strides=(2, 2),
				filters=256, padding='same')(conv1_out)
			conv2_out = relu(BatchNormalization()(conv2_out))
			conv3_out = Conv2D(
				kernel_size=(3, 3), strides=(2, 2),
				filters=256, padding='same')(conv2_out)
			conv3_out = relu(BatchNormalization()(conv3_out))
			conv4_out = Conv2D(
				kernel_size=(3, 3), strides=(2, 2),
				filters=256)(conv3_out)
			conv4_out = relu(BatchNormalization()(conv4_out))
			dec1_out = Conv2D(
				kernel_size=(1, 1), strides=(1, 1),
				filters=128)(conv4_out)
			dec1_out = relu(BatchNormalization()(dec1_out))
			bs_out = self.bilinear_upsampler(conv4_out, conv3_out.shape[1:3])
			conv5_out = Conv2D(
				kernel_size=(3, 3), strides=(1, 1),
				filters=256)(bs_out)
			conv5_out = relu(BatchNormalization()(conv5_out))
			conv5_out = conv3_out +
					self.bilinear_upsampler(conv5_out, conv3_out.shape[1:3])
			dec2_out = Conv2D(
				kernel_size=(1, 1), strides=(1, 1),
				filters=128)(conv5_out)
			dec2_out = relu(BatchNormalization()(dec2_out))
			conv6_out = Conv2D(
				kernel_size=(3, 3), strides=(1, 1),
				filters=256)(conv5_out)
			conv6_out = relu(BatchNormalization()(conv6_out))
			conv6_out = conv2_out +
					self.bilinear_upsampler(conv6_out, conv2_out.shape[1:3])
			dec3_out = Conv2D(
				kernel_size=(1, 1), strides=(1, 1),
				filters=128)(conv6_out)
			dec3_out = relu(BatchNormalization()(dec3_out))
			conv7_out = Conv2D(
				kernel_size=(3, 3), strides=(1, 1),
				filters=256)(conv6_out)
			conv7_out = relu(BatchNormalization()(conv7_out))
			conv7_out = conv1_out +
					self.bilinear_upsampler(conv7_out, conv1_out.shape[1:3])
			dec4_out = Conv2D(
				kernel_size=(1, 1), strides=(1, 1),
				filters=128)(conv7_out)
			dec4_out = relu(BatchNormalization()(dec4_out))
			conv8_out = Conv2D(
				kernel_size=(3, 3), strides=(1, 1),
				filters=256)(conv7_out)
			conv8_out = relu(BatchNormalization()(conv8_out))
			conv8_out = conv0_out +
					self.bilinear_upsampler(conv8_out, conv0_out.shape[1:3])
			dec5_out = Conv2D(
				kernel_size=(1, 1), strides=(1, 1),
				filters=128)(conv8_out)
			dec5_out = relu(BatchNormalization()(dec5_out))
			conv9_out = Conv2D(
				kernel_size=(3, 3), strides=(1, 1),
				filters=256)(conv8_out)
			conv9_out = relu(BatchNormalization()(conv9_out))
			conv9_out = self.features +
					self.bilinear_upsampler(conv9_out, self.features.shape[1:3])
			dec6_out = Conv2D(
				kernel_size=(1, 1), strides=(1, 1),
				filters=128)(conv9_out)
			dec6_out = relu(BatchNormalization()(dec6_out))
		"""
