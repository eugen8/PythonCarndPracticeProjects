
import tensorflow as tf
import numpy as np

# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [1, 2, 2 ],
    [1, 0, 0 ],
], dtype=np.float32)

x = np.array([
    [[2,3,4], [2,3,4], [2,3,4] ],
    [[1,1,1], [0,0,1], [7,0,0] ],
], dtype=np.float32)


# print(x)

with tf.Session() as sess:
  with tf.device("/cpu:0"):
      normal = tf.nn.l2_normalize(x,dim=(0,1))
      normal0 = tf.nn.l2_normalize(x, dim=(0))
      normal1 = tf.nn.l2_normalize(x, dim=1)
      normal2 = tf.nn.l2_normalize(x, dim=2)

      res = sess.run(normal)
      res0 = sess.run(normal0)
      res1 = sess.run(normal1)
      res2 = sess.run(normal2)
      res012 = sess.run(tf.nn.l2_normalize(x, dim=(0,1,2)))
      print(res, "  (0,1)------  \r\n")
      print(res0, "  (0)------\r\n  ")
      print(res1, "  (1)------  ")
      print(res2, "  (2)------  ")
      print(res012, "  (0,1,2)------  ")


a = x[0,:]
a = a.reshape((1,-1))
sumsq = sum(a.T * a.T)
# print( sumsq, 4/25 )



m=sum(x[1:])
