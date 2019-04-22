import sys
import json
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np
from m2det import FFM
from m2det import TUM
from m2det import SFAM
tf.enable_eager_execution()

config_file = sys.argv[1]
assert config_file, "Specify config file"
config = json.load(open(config_file, "r"))

f1 = tf.Variable(np.random.rand(1, 40, 40, 512), dtype=tf.float32)
f2 = tf.Variable(np.random.rand(1, 20, 20, 1024), dtype=tf.float32)
f3 = tf.Variable(np.random.rand(1, 40, 40, 128), dtype=tf.float32)


ffm = FFM(f1, f2)

#collecting decoder outputs from TUMs
decoder_outs = []
for i in range(config["model"]["tums_no"]):
	if i == 0:
		features = ffm.v1()
		features = Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=256)(features)
	else:
		features = ffm.v2(decoder_outs[i-1][-1])
	decoder_outs.append(TUM(config,features).forward())

#constructing mlfpn using SFAM
mlfpn = SFAM(config, decoder_outs).forward()
