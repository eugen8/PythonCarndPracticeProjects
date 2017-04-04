# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session
crossEntropy = - tf.reduce_sum( tf.mul(one_hot , tf.log(softmax)) )
# crossEntropy = -tf.reduce_sum(one_hot * tf.log(softmax))

with tf.Session() as sess:
    print(tf.Tensor.eval(one_hot, feed_dict={one_hot: one_hot_data, softmax: softmax_data}))
    print(tf.Tensor.eval(softmax, feed_dict={one_hot: one_hot_data, softmax: softmax_data}))
    print(tf.Tensor.eval(one_hot * tf.log(softmax), feed_dict={one_hot: one_hot_data, softmax: softmax_data}))
    print(tf.Tensor.eval(  tf.reduce_sum(one_hot * tf.log(softmax)), feed_dict={one_hot: one_hot_data, softmax: softmax_data}))
    print(tf.Tensor.eval(-tf.reduce_sum(one_hot * tf.log(softmax)),
        feed_dict={one_hot: one_hot_data, softmax: softmax_data}))
    output = sess.run(crossEntropy,
                      feed_dict={one_hot: one_hot_data, softmax:softmax_data})
    print(output)
