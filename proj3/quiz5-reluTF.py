# Solution is available in the other "solution.py" tab
import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model
biasespl = tf.placeholder(tf.float32)

hidden_features = tf.add( tf.matmul(features, weights[0]),biases[0])
# hidden_features = tf.add( tf.matmul(features, weights[0]),biasespl)
hidden_features = tf.nn.relu(hidden_features)
output_model = tf.add( tf.matmul(hidden_features, weights[1]),biases[1] )

# TODO: Print session results
with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    output = session.run(output_model)
    # output = session.run(output_model, feed_dict={biasespl:biases[0]})
print(output)
"""V1:
[[  5.11000013   8.44000053]
 [  0.           0.        ]
 [ 24.01000214  38.23999786]]"""
