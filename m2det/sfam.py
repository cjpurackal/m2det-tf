import tensorflow as tf
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import Conv2D, Dense

class SFAM:
	def __init__(self, config, decoder_outs):
		self.config = config
		self.decoder_outs = decoder_outs

	def forward(self):
		mlfpn = []
		for i in range(self.config["model"]["scales"]+1):
			feature_cube = tf.concat([self.decoder_outs[j][i] for j in range(self.config["model"]["tums_no"])], axis=3)
			attention = tf.reduce_mean(feature_cube, axis=[1, 2], keepdims=True)
			attention = Dense(units=64,activation=relu)(attention)
			attention = Dense(units=1024,activation=sigmoid)(attention)
			feature_cube = feature_cube * attention
			mlfpn.append(feature_cube)

		return mlfpn