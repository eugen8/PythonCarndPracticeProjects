import tensorflow as tf
import numpy as np

# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))

print(x)

with tf.Session() as sess:
  with tf.device("/cpu:0"):
      matrix1 = tf.constant([[3., 3.]])
      matrix2 = tf.constant([[2.], [2.]])
      product = tf.matmul(matrix1, matrix2)
      print(product)
      p = sess.run(product)
      print(p)
