import keras
import argparse
import os
import tensorflow as tf
import cv2
import numpy as np

from pprint import pprint
from keras.backend.tensorflow_backend import set_session
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.models import load_model
from config import *


def build_model(flags, model_file):
    if flags.model_name == "inception_v3":
        base_model = InceptionV3(input_shape=(flags.image_size, flags.image_size, 3), include_top=False,
                                 weights='imagenet')
    elif flags.model_name == "inception_resnet_v2":
        base_model = InceptionResNetV2(input_shape=(flags.image_size, flags.image_size, 3), include_top=False,
                                       weights='imagenet')
    elif flags.model_name == "xception":
        base_model = Xception(input_shape=(flags.image_size, flags.image_size, 3), include_top=False,
                              weights='imagenet')
    else:
        raise NotImplementedError("Not a supported model: {}".format(flags.model_name))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dropout(0.25)(x)
    x = Dense(1, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=x)

    if os.path.isfile(model_file):
        print "[x] loading model file {}".format(model_file)
        model.load_weights(model_file)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="model name: inception_v3 ....", default="inception_v3")
    parser.add_argument('--exp', type=str, help="experiment name", default="ENH")
    parser.add_argument('--sex', type=int, help="1 for male, 2 for female", default=1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--inferenced_image', type=str, default=RSNA_TRAIN_DATA+"/1451.png")
    FLAGS = parser.parse_args()

    pprint(FLAGS)

    print "[x] building models on GPU {}".format(FLAGS.gpu_id)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(FLAGS.gpu_id)

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    set_session(tf.Session(config=config))

    output_path = "E{}_M{}_S{}_I{}".format(FLAGS.exp, FLAGS.model_name, FLAGS.sex, FLAGS.image_size)

    model = build_model(FLAGS, output_path+"/weights.h5")
    print "build model done"

    if FLAGS.model_name == "inception_v3":
        from keras.applications.inception_v3 import preprocess_input
    elif FLAGS.model_name == "inception_resnet_v2":
        from keras.applications.inception_resnet_v2 import preprocess_input
    elif FLAGS.model_name == "xception":
        from keras.applications.xception import preprocess_input
    else:
        raise ValueError("Not support {}".format(FLAGS.model_name))

    print "load image file"
    img = cv2.imread(FLAGS.inferenced_image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (FLAGS.image_size, FLAGS.image_size))
    img = preprocess_input(np.array(img, dtype=np.float32))
    print "image shape = {}".format(img.shape)

    val = model.predict(np.expand_dims(img, 0))
    print "->{}".format(np.squeeze(val))
    """
    if FLAGS.sex == 1:
        val = val * 84.2863236515 + 135.30367335
    elif FLAGS.sex == 2:
        val = val * 75.8162237968 + 117.880235376
    print "[xxx]" + val
    """
