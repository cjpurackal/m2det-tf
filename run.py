import tensorflow as tf
import numpy as np
from m2det import FFM

tf.enable_eager_execution()

f1 = tf.Variable(np.random.rand(1, 40, 40, 512))
f2 = tf.Variable(np.random.rand(1, 20, 20, 1024))
f3 = tf.Variable(np.random.rand(1, 40, 40, 128))


ffm = FFM(f1, f2)

v1_out = ffm.v1()
print (v1_out.shape)
v2_out = ffm.v2(f3)
print(v2_out.shape)