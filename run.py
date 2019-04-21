import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np
from m2det import FFM
from m2det import TUM
tf.enable_eager_execution()

f1 = tf.Variable(np.random.rand(1, 40, 40, 512), dtype=tf.float32)
f2 = tf.Variable(np.random.rand(1, 20, 20, 1024), dtype=tf.float32)
f3 = tf.Variable(np.random.rand(1, 40, 40, 128), dtype=tf.float32)

tums_no = 3

ffm = FFM(f1, f2)

decoder_outs = []
for i in range(tums_no):
	if i == 0:
		features = ffm.v1()
		features = Conv2D(kenel_size=(1, 1), strides=(1, 1), filters=256)(features)
	else:
		features = ffm.v2(f3)

	decoder_outs.append(TUM(features))

v1_out = ffm.v1()
v2_out = ffm.v2(f3)

TUM(v1_out)