import tensorflow as tf

SMOOTH = 1.
class BatchNorm(object):

    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv2d(input_, output_dim, k=3, s=1, stddev=0.02, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, s, s, 1], padding="SAME")
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(value=0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


def deconv2d(input_, output_dim, k=3, s=1, stddev=0.02, name='deconv2d'):
    input_shape = input_.get_shape().as_list()
    output_shape = [input_shape[0], input_shape[1]*2, input_shape[2]*2, output_dim]
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, s, s, 1])
        biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(value=0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv


def upsampling2d(input_, output_dim, k=3, s=1, stddev=0.02, name='upsampling2d'):
    input_shape = input_.get_shape().as_list()
    output_shape = [input_shape[0], input_shape[1] * s, input_shape[2] * s, output_dim]
    with tf.variable_scope(name):
        w = tf.get_variable("w", [k, k, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        conv2d_trans = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                              strides=[1, s, s, 1])
        biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(value=0.0))
        conv2d_trans = tf.reshape(tf.nn.bias_add(conv2d_trans, biases), conv2d_trans.get_shape())
        return conv2d_trans


def maxpool2d(input_, k=2, s=2):
    return tf.nn.max_pool(input_, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding="SAME")


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias= tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix)


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)


def relu(x):
    return tf.nn.relu(x)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def dropout(x, keep_prob=0.5):
    return tf.nn.dropout(x, keep_prob)


def dice_coef_loss(prob, label):
    batch_size, h, w, n_class = label.get_shape().as_list()
    assert n_class == 1
    flat_label = tf.reshape(label, [-1, h*w*n_class])
    flat_prob = tf.reshape(prob, [-1, h*w*n_class])
    intersection = tf.reduce_mean(2*tf.multiply(flat_prob, flat_label))+SMOOTH
    union = tf.reduce_mean(tf.add(flat_prob, flat_label))+SMOOTH
    loss = 1 - tf.div(intersection, union)
    return loss


def pixelwise_cross_entropy(logit, label):
    flat_logit = tf.reshape(logit, [-1])
    flat_label = tf.reshape(label, [-1])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logit, labels=flat_label))
    return loss