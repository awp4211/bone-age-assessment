import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from keras.backend.tensorflow_backend import set_session
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_input
from keras.models import Model
from keras import optimizers
from keras.layers import GlobalAveragePooling2D, Dense
from tqdm import tqdm
import keras

from utils_data import load_sex_ids, load_obj, load_data_sex
from config import *
from step4_cam import inference

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help="experiment name", default="CAM")
parser.add_argument('--img_size', type=int, help="image size", default=128)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--batch_size', type=int, help="training batch size", default=16)
parser.add_argument('--sex', type=int, help="0 for all, 1 for male, 2 for female", default=1)
parser.add_argument('--start_layer', type=int, help="start_layer", default=-1)
FLAGS = parser.parse_args()

exp_name = FLAGS.exp
gpu_id = FLAGS.gpu_id
img_size = FLAGS.img_size
batch_size = FLAGS.batch_size
sex = FLAGS.sex
start_layer = FLAGS.start_layer


model_out_dir = "E{}_S{}_IMG_{}/model".format(exp_name, sex, img_size)
inference_out_dir = "E{}_S{}_IMG_{}/cam".format(exp_name, sex, img_size)

input_shape = (img_size, img_size, 3)
base_model = Xception(input_shape=input_shape, weights="imagenet", include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
base_model.load_weights(filepath=model_out_dir + "/base_model.h5")
predictions = Dense(1, activation=keras.activations.relu)(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(model_out_dir+"/model.h5")
optimizer = optimizers.RMSprop(lr=5E-4, decay=0.95)
model.compile(optimizer=optimizer, loss='mean_absolute_error')


print "[x] building models on GPU {}".format(gpu_id)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

print "load model and data"
data_ids = load_sex_ids(sex)
data_x, data_y = load_data_sex(sex, img_size, xception_input)
weights_history = load_obj(model_out_dir+"/weights.pkl")

print "inference"
inference(data_ids, data_y, base_model, weights_history, img_size)
