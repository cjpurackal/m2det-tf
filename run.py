import tensorflow as tf
import numpy as np
from m2det import FFM
from m2det import TUM
tf.enable_eager_execution()

f1 = tf.Variable(np.random.rand(1, 40, 40, 512), dtype=tf.float32)
f2 = tf.Variable(np.random.rand(1, 20, 20, 1024), dtype=tf.float32)
f3 = tf.Variable(np.random.rand(1, 40, 40, 128), dtype=tf.float32)


ffm = FFM(f1, f2)

v1_out = ffm.v1()
v2_out = ffm.v2(f3)

TUM(v2_out)