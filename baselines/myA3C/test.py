import tensorflow as tf
import numpy as np


sess = tf.Session()

a = [[1], [5]]
print np.array(a).shape
b = tf.zeros(2)
cc = a + b

c = tf.one_hot(a, depth=6, on_value=1, off_value=0)
aa = tf.reduce_sum(a, 0)
print sess.run(cc)