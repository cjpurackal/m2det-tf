import sys
import json
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np
from m2det.feature_extractors import VGG16, Darknet21
from m2det import FFM
from m2det import TUM
from m2det import SFAM
from m2det.utils import bilinear_upsampler
# tf.enable_eager_execution()


config_file = sys.argv[1]
assert config_file, "Specify config file"
config = json.load(open(config_file, "r"))

image = tf.Variable(np.random.rand(1,320,320, 3), dtype=tf.float32)
# f1, f2 = Darknet21(image, config).forward()
f1, f2 = VGG16(image, config).forward()
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

# for cube in mlfpn:
# 	print (cube.shape)