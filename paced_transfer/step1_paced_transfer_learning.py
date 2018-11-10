import numpy as np
import pandas as pd
import cv2
import argparse
import os
import pprint
import pickle
import keras
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K

from keras.utils import multi_gpu_model
from keras import optimizers
from keras.backend.tensorflow_backend import set_session

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_input


from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from tqdm import tqdm

from config import *
from utils_training import _set_trainable_layers, _pprint
from utils_data import save_obj
from utils_data import load_sex_ids, load_data
from utils_metric import regression_metric


def _build_regressor(img_size=299, num_gpu=1, start_layer=-1, model_file=None, learning_rate=1E-4):
    input_shape = (img_size, img_size, 3)
    if start_layer == -1:
        base_model = Xception(input_shape=input_shape, weights="imagenet", include_top=False)
        _set_trainable_layers(base_model, len(base_model.layers))
    else:
        base_model = Xception(input_shape=input_shape, weights=None, include_top=False)
        _set_trainable_layers(base_model, start_layer)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation=keras.activations.relu)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = optimizers.RMSprop(lr=learning_rate, decay=0.95)
    if num_gpu > 1:
        model = multi_gpu_model(model, num_gpu)
    print "[x] compile model on %d GPU(s)" % num_gpu

    if model_file!=None:
        model.load_weights(model_file)

    model.compile(optimizer=optimizer, loss="mean_squared_error")
    model.summary(print_fn=_pprint)
    return model


def train(n_epoch=N_TRAINING_EPOCH, img_size=299, sex=0, batch_size=16, num_gpu=1, start_layer=-1, start_epoch=0):
    assert start_layer in [-1, XCEPTION_EXIT_START, XCEPTION_MID_START, XCEPTION_ENTRY_START, 0]
    assert sex in [0, 1, 2]
    # model file path
    if start_layer != -1:
        model_file = model_out_dir+"/model.h5"
    else:
        model_file = None

    # learning rate
    if start_layer == -1:
        learning_rate = 1E-2
    elif start_layer == XCEPTION_EXIT_START:
        learning_rate = 1E-3
    elif start_layer == XCEPTION_MID_START:
        learning_rate = 5E-4
    elif start_layer == XCEPTION_ENTRY_START:
        learning_rate = 1E-4
    else:
        learning_rate = 5E-5

    model = _build_regressor(img_size, num_gpu, start_layer, model_file, learning_rate)

    best_mae = np.inf
    tolerance = 0

    data_ids = load_sex_ids(sex)

    for epoch in tqdm(range(start_epoch+1, start_epoch + n_epoch+1)):
        print "[x] epoch {} -------------------------------------------".format(epoch)
        for mini_batch in range(len(data_ids)//batch_size):
            batch_x, batch_y = load_data(sex=sex, img_size=img_size, batch_size=batch_size, augment_times=7)
            loss = model.train_on_batch(x=batch_x, y=batch_y)
            if mini_batch % 50 == 0:
                print "--epoch {}, mini_batch {}, loss {}".format(epoch, mini_batch, loss)

        # test
        print "[x] test in epoch {}".format(epoch)
        losses = 0.0
        for mini_batch in range(int(0.2*len(data_ids)//batch_size)):
            batch_x, batch_y = load_data(sex=sex, img_size=img_size, batch_size=batch_size, augment_times=0)
            loss = model.test_on_batch(batch_x, batch_y)
            losses += loss
        losses = losses/(int(0.3*len(data_ids)//batch_size))
        print "== epoch {}, test loss {}".format(epoch, losses)

        # test and metric
        print "[x] predict in epoch {}".format(epoch)
        y_true = []
        y_pred = []
        for mini_batch in range(int(0.2*len(data_ids)//batch_size)):
            batch_x, batch_y = load_data(sex=sex, img_size=img_size, batch_size=batch_size, augment_times=0)
            pred_y = model.predict_on_batch(batch_x)
            for i in range(batch_size):
                y_true.append(batch_y[i]*SCALE)
                y_pred.append(pred_y[i]*SCALE)

        evs, mae, mse, meae, r2s, ccc = regression_metric(np.array(y_true), np.array(y_pred))
        save_obj({"evs": evs, "mae": mae, "mse": mse, "meae": meae, "r2s": r2s, "ccc": ccc, "loss": losses},
                 name=metric_out_dir+"/epoch_{}.pkl".format(epoch))

        if mae < best_mae:
            best_mae = mae
            tolerance = 0
            model.save_weights(model_out_dir + "/model.h5")
        else:
            tolerance += 1

        print "[x] epoch {}, evs {}, mae {}, mse {}, meae {}, r2s {}, ccc {}".format(epoch, evs, mae, mse, meae, r2s, ccc)

        if tolerance > TOLERANCE:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help="experiment name", default="regression")
    parser.add_argument('--img_size', type=int, help="image size", default=128)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, help="training batch size", default=4)
    parser.add_argument('--sex', type=int, help="0 for all, 1 for male, 2 for female", default=1)
    parser.add_argument('--start_layer', type=int, help="start_layer", default=-1)
    FLAGS = parser.parse_args()

    pprint.pprint(FLAGS)

    exp_name = FLAGS.exp
    num_gpu = 1
    gpu_id = FLAGS.gpu_id
    img_size = FLAGS.img_size
    batch_size = FLAGS.batch_size * num_gpu
    sex = FLAGS.sex
    start_layer = FLAGS.start_layer

    print "[x] building models on GPU {}".format(gpu_id)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    metric_out_dir = "E{}_S{}_IMG_{}/metric".format(exp_name, sex, img_size)
    model_out_dir = "E{}_S{}_IMG_{}/model".format(exp_name, sex, img_size)
    if not os.path.isdir(metric_out_dir):
        os.makedirs(metric_out_dir)
    if not os.path.isdir(model_out_dir):
        os.makedirs(model_out_dir)
    start_epoch = len(os.listdir(metric_out_dir))

    # training
    train(n_epoch=N_TRAINING_EPOCH, img_size=img_size, sex=sex, batch_size=batch_size,
          num_gpu=num_gpu, start_layer=start_layer, start_epoch=start_epoch)
