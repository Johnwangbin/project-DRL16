import tensorflow as tf



sess = tf.Session()

a = [[1,2,4], [5,2,1]]

c = tf.one_hot(a, depth=6, on_value=1, off_value=0)

print sess.run(c)[0]