import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.activations import relu


class TUM:
	def __init__(self, features):
		self.features = features
		self.forward()

	def bilinear_upsampler(self, tensor, shape):
		return tf.image.resize_images(tensor, shape)

	def forward(self):
		#encoder
		conv0_out = Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=256, padding='same')(self.features)
		conv0_out = relu(BatchNormalization()(conv0_out))
		
		conv1_out = Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=256, padding='same')(conv0_out)
		conv1_out = relu(BatchNormalization()(conv1_out))
		
		conv2_out = Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=256, padding='same')(conv1_out)
		conv2_out = relu(BatchNormalization()(conv2_out))
		
		conv3_out = Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=256, padding='same')(conv2_out)
		conv3_out = relu(BatchNormalization()(conv3_out))
		
		conv4_out = Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=256)(conv3_out)
		conv4_out = relu(BatchNormalization()(conv4_out))

		#decoder
		dec1_out = Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=128)(conv4_out)
		dec1_out = relu(BatchNormalization()(dec1_out))

		bs_out = self.bilinear_upsampler(conv4_out, conv3_out.shape[1:3])

		conv5_out = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=256)(bs_out)
		conv5_out = relu(BatchNormalization()(conv5_out))		
		conv5_out = conv3_out + self.bilinear_upsampler(conv5_out, conv3_out.shape[1:3])

		dec2_out = Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=128)(conv5_out)
		dec2_out = relu(BatchNormalization()(dec2_out))

		conv6_out = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=256)(conv5_out)
		conv6_out = relu(BatchNormalization()(conv6_out))		
		conv6_out = conv2_out + self.bilinear_upsampler(conv6_out, conv2_out.shape[1:3])

		dec3_out = Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=128)(conv6_out)
		dec3_out = relu(BatchNormalization()(dec3_out))

		conv7_out = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=256)(conv6_out)
		conv7_out = relu(BatchNormalization()(conv7_out))		
		conv7_out = conv1_out + self.bilinear_upsampler(conv7_out, conv1_out.shape[1:3])

		dec4_out = Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=128)(conv7_out)
		dec4_out = relu(BatchNormalization()(dec4_out))

		conv8_out = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=256)(conv7_out)
		conv8_out = relu(BatchNormalization()(conv8_out))		
		conv8_out = conv0_out + self.bilinear_upsampler(conv8_out, conv0_out.shape[1:3])

		dec5_out = Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=128)(conv8_out)
		dec5_out = relu(BatchNormalization()(dec5_out))

		conv9_out = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=256)(conv8_out)
		conv9_out = relu(BatchNormalization()(conv9_out))		
		conv9_out = self.features + self.bilinear_upsampler(conv9_out, self.features.shape[1:3])

		dec6_out = Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=128)(conv9_out)
		dec6_out = relu(BatchNormalization()(dec6_out))

		