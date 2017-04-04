# Solution is available in the other "solution.py" tab
import tensorflow as tf

weights = tf.Variable(tf.random_normal([3, 2], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([2]), name="biases")

z = tf.zeros([3,2])
zv = tf.Variable(z)
zv = tf.add(zv, 8)

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)

    x = tf.Tensor.eval(tf.truncated_normal([4,2],stddev=0.75))
    print('truncated_normal', x, type(x))

    y = tf.Tensor.eval(weights)
    print('weights variable',y, type(y), 'from', weights)

    print(zv, tf.Tensor.eval(zv))
    print("Multiplying y to zv",tf.Tensor.eval(tf.matmul(weights, zv, transpose_b=True)))