import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim

from config import *
from tf_ops import *

# generate Model for U-Net
def unet(x, is_train, n_filters, name="unet_v2"):
    k = 3
    s = 2
    tensor_shape = x.get_shape().as_list()
    img_ch = 1
    out_ch = 1
    assert len(tensor_shape) == 4  # batch, d, h, w, ch
    assert tensor_shape[-1] == out_ch

    if DEBUG_MODEL:
        print "x shape = {}".format(x.get_shape())

    with tf.variable_scope(name) as scope:
        conv1 = conv2d(x, n_filters, k=k, name="conv1_0")
        bn1_0 = BatchNorm(name="bn1_0")
        conv1 = bn1_0(conv1, train=is_train)
        conv1 = relu(conv1)
        if DEBUG_MODEL: print "conv1_0 shape = {}".format(conv1.get_shape())
        conv1 = conv2d(conv1, n_filters, k=k, name="conv1_1")
        bn1_1 = BatchNorm(name="bn1_1")
        conv1 = bn1_1(conv1, train=is_train)
        conv1 = relu(conv1)
        if DEBUG_MODEL: print "conv1_1 shape = {}".format(conv1.get_shape())
        pool1 = maxpool2d(conv1, k=2, s=s)
        if DEBUG_MODEL: print "pool1 shape = {}".format(pool1.get_shape())

        conv2 = conv2d(pool1, 2*n_filters, k=k, name="conv2_0")
        bn2_0 = BatchNorm(name="bn2_0")
        conv2 = bn2_0(conv2, train=is_train)
        conv2 = relu(conv2)
        if DEBUG_MODEL: print "conv2_0 shape = {}".format(conv2.get_shape())
        conv2 = conv2d(conv2, 2*n_filters, k=k, name="conv2_1")
        bn2_1 = BatchNorm(name="bn2_1")
        conv2 = bn2_1(conv2, train=is_train)
        conv2 = relu(conv2)
        if DEBUG_MODEL: print "conv2_1 shape = {}".format(conv2.get_shape())
        pool2 = maxpool2d(conv2, k=2, s=s)
        if DEBUG_MODEL: print "pool2 shape = {}".format(pool2.get_shape())

        conv3 = conv2d(pool2, 4*n_filters, k=k, name="conv3_0")
        bn3_0 = BatchNorm(name="bn3_0")
        conv3 = bn3_0(conv3, train=is_train)
        conv3 = relu(conv3)
        if DEBUG_MODEL: print "conv3_0 shape = {}".format(conv3.get_shape())
        conv3 = conv2d(conv3, 4*n_filters, k=k, name="conv3_1")
        bn3_1 = BatchNorm(name="bn3_1")
        conv3 = bn3_1(conv3, train=is_train)
        conv3 = relu(conv3)
        if DEBUG_MODEL: print "conv3_1 shape = {}".format(conv3.get_shape())
        pool3 = maxpool2d(conv3, k=2, s=s)
        if DEBUG_MODEL: print "pool3 shape = {}".format(pool3.get_shape())

        conv4 = conv2d(pool3, 8*n_filters, k=k, name="conv4_0")
        bn4_0 = BatchNorm(name="bn4_0")
        conv4 = bn4_0(conv4, train=is_train)
        conv4 = relu(conv4)
        if DEBUG_MODEL: print "conv4_0 shape = {}".format(conv4.get_shape())
        conv4 = conv2d(conv4, 8*n_filters, k=k, name="conv4_1")
        bn4_1 = BatchNorm(name="bn4_1")
        conv4 = bn4_1(conv4, train=is_train)
        conv4 = relu(conv4)
        if DEBUG_MODEL: print "conv4_1 shape = {}".format(conv4.get_shape())
        pool4 = maxpool2d(conv4, k=2, s=s)
        if DEBUG_MODEL: print "pool4 shape = {}".format(pool4.get_shape())

        conv5 = conv2d(pool4, 16*n_filters, k=k, name="conv5_0")
        bn5_0 = BatchNorm(name="bn5_0")
        conv5 = bn5_0(conv5, train=is_train)
        conv5 = relu(conv5)
        if DEBUG_MODEL: print "conv5_0 shape = {}".format(conv5.get_shape())
        conv5 = conv2d(conv5, 16*n_filters, k=k, name="conv5_1")
        bn5_1 = BatchNorm(name="bn5_1")
        conv5 = bn5_1(conv5, train=is_train)
        conv5 = relu(conv5)
        if DEBUG_MODEL: print "conv5_1 shape = {}".format(conv5.get_shape())

        up1 = upsampling2d(conv5, 16*n_filters, k=k, s=2, name="up1")
        if DEBUG_MODEL: print "up1 before concat shape = {}".format(up1.get_shape())
        up1 = tf.concat([up1, conv4], axis=3)
        if DEBUG_MODEL: print "up1 concat shape = {}".format(up1.get_shape())
        conv6 = conv2d(up1, 8*n_filters, k=k, name="conv6_0")
        bn6_0 = BatchNorm(name="bn6_0")
        conv6 = bn6_0(conv6, train=is_train)
        conv6 = relu(conv6)
        if DEBUG_MODEL: print "conv6_0 shape = {}".format(conv6.get_shape())
        conv6 = conv2d(conv6, 8*n_filters, k=k, name="conv6_1")
        bn6_1 = BatchNorm(name="bn6_1")
        conv6 = bn6_1(conv6, train=is_train)
        conv6 = relu(conv6)
        if DEBUG_MODEL: print "conv6_1 shape = {}".format(conv6.get_shape())

        up2 = upsampling2d(conv6, 8*n_filters, k=k, s=2, name="up2")
        if DEBUG_MODEL: print "up2 before concat shape = {}".format(up2.get_shape())
        up2 = tf.concat([up2, conv3], axis=3)
        if DEBUG_MODEL: print "up2 concat shape = {}".format(up2.get_shape())
        conv7 = conv2d(up2, 4*n_filters, k=k, name="conv7_0")
        bn7_0 = BatchNorm(name="bn7_0")
        conv7 = bn7_0(conv7, train=is_train)
        conv7 = relu(conv7)
        if DEBUG_MODEL: print "conv7_0 shape = {}".format(conv7.get_shape())
        conv7 = conv2d(conv7, 4*n_filters, k=k, name="conv7_1")
        bn7_1 = BatchNorm(name="bn7_1")
        conv7 = bn7_1(conv7, train=is_train)
        conv7 = relu(conv7)
        if DEBUG_MODEL: print "conv7_1 shape = {}".format(conv7.get_shape())

        up3 = upsampling2d(conv7, 4*n_filters, k=k, s=2, name="up3")
        if DEBUG_MODEL: print "up3 before concat shape = {}".format(up3.get_shape())
        up3 = tf.concat([up3, conv2], axis=3)
        if DEBUG_MODEL: print "up3 concat shape = {}".format(up3.get_shape())
        conv8 = conv2d(up3, 2*n_filters, k=k, name="conv8_0")
        bn8_0 = BatchNorm(name="bn8_0")
        conv8 = bn8_0(conv8, train=is_train)
        conv8 = relu(conv8)
        if DEBUG_MODEL: print "conv8_0 shape = {}".format(conv8.get_shape())
        conv8 = conv2d(conv8, 2*n_filters, k=k, name="conv8_1")
        bn8_1 = BatchNorm(name="bn8_1")
        conv8 = bn8_1(conv8, train=is_train)
        conv8 = relu(conv8)
        if DEBUG_MODEL: print "conv8_1 shape = {}".format(conv8.get_shape())

        up4 = upsampling2d(conv8, 2*n_filters, k=k, s=2, name="up4")
        if DEBUG_MODEL: print "up4 before concat shape = {}".format(up4.get_shape())
        up4 = tf.concat([up4, conv1], axis=3)
        if DEBUG_MODEL: print "up4 concat shape = {}".format(up4.get_shape())
        conv9 = conv2d(up4, n_filters, k=k, name="conv9_0")
        bn9_0 = BatchNorm(name="bn9_0")
        conv9 = bn9_0(conv9, train=is_train)
        conv9 = relu(conv9)
        if DEBUG_MODEL: print "conv9_0 shape = {}".format(conv9.get_shape())
        conv9 = conv2d(conv9, n_filters, k=k, name="conv9_1")
        bn9_1 = BatchNorm(name="bn9_1")
        conv9 = bn9_1(conv9, train=is_train)
        conv9 = relu(conv9)
        if DEBUG_MODEL: print "conv9_1 shape = {}".format(conv9.get_shape())

        outputs = conv2d(conv9, out_ch, k=1, name="conv10")
        outputs = sigmoid(outputs)
        if DEBUG_MODEL: print "outputs shape = {}".format(outputs.get_shape())

        return outputs, conv9
