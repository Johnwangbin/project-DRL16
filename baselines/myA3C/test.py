import tensorflow as tf
import numpy as np
import threading
import logging
import time
# sess = tf.Session()
#
a = [[1, 3], [5, 3]]
print np.array(a).shape
b = tf.zeros(2)
cc = a + b



c = tf.one_hot(a, depth=6, on_value=1, off_value=0)
aa = np.reshape(a, [-1])
# print sess.run(aa)
print aa
# logging.basicConfig(level=logging.DEBUG,
#                     format='(%(threadName)-10s) %(message)s',
#                     )
#
# def worker(num):
#     print 'worker : %s' % num
#     logging.debug('Starting')
#     if num == 0:
#         time.sleep(2)
#     logging.debug('Exiting')
#     return
#
# threads = []
#
# for i in range(2):
#     t = threading.Thread(target=worker, args=(i,))
#     if i == 0:
#         t.setDaemon(True)  # the main thread can exit when this thread is still running.
#     threads.append(t)
#     t.start()


# a_dist = [[0.3,  0.2,  0.1, 0.15, 0.15, 0.1]]
#
# actions = np.arange(len(a_dist[0]))
# np.random.seed(1024)
# aa = np.random.choice(actions, p=a_dist[0])
# np.random.seed(1024)
# a = np.random.choice(a_dist[0], p=a_dist[0])
# a = np.argmax(a_dist == a)
#
# print a, aa