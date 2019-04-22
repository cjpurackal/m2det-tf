import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.activations import relu, sigmoid
import numpy as np
from m2det import FFM
from m2det import TUM
tf.enable_eager_execution()

f1 = tf.Variable(np.random.rand(1, 40, 40, 512), dtype=tf.float32)
f2 = tf.Variable(np.random.rand(1, 20, 20, 1024), dtype=tf.float32)
f3 = tf.Variable(np.random.rand(1, 40, 40, 128), dtype=tf.float32)

tums_no = 3
scales = 5

ffm = FFM(f1, f2)

decoder_outs = []
for i in range(tums_no):
	if i == 0:
		features = ffm.v1()
		features = Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=256)(features)
	else:
		features = ffm.v2(decoder_outs[i-1][-1])

	decoder_outs.append(TUM(features).forward())

mlfpn = []
for i in range(scales+1):
	feature_cube = tf.concat([decoder_outs[j][i] for j in range(tums_no)], axis=3)
	attention = tf.reduce_mean(feature_cube, axis=[1, 2], keepdims=True)
	attention = Dense(units=64,activation=relu)(attention)
	attention = Dense(units=1024,activation=sigmoid)(attention)
	feature = feature_cube * attention
	# for j in range(tums_no):

	# 	mlfpn.append(decoder_outs[j][i])

# v1_out = ffm.v1()
# v2_out = ffm.v2(f3)
# do = TUM(v2_out).forward()
# print (do[-1].shape)