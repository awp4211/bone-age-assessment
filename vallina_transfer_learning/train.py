import keras
import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.applications.xception import Xception

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers
from pprint import pprint
from tqdm import tqdm

from config import *
from utils import *
from utils_metric import regression_metric
from utils import _pprint


def _build_regressor(img_size=299, learning_rate=1E-4):
    input_shape = (img_size, img_size, 3)
    base_model = Xception(input_shape=input_shape, weights="imagenet", include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation=keras.activations.relu)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = optimizers.RMSprop(lr=learning_rate, decay=0.95)

    model.compile(optimizer=optimizer, loss="mean_squared_error")
    model.summary(print_fn=_pprint)
    return model


def train(n_epoch, img_size=299, sex=0, batch_size=16, learning_rate=1E-4):
    assert sex in [0, 1, 2]
    model = _build_regressor(img_size, learning_rate)

    best_mae = np.inf

    data_ids = load_sex_ids(sex)

    for epoch in tqdm(range(n_epoch)):
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
            model.save_weights(model_out_dir + "/model.h5")

        print "[x] epoch {}, evs {}, mae {}, mse {}, meae {}, r2s {}, ccc {}".format(epoch, evs, mae, mse, meae, r2s, ccc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help="experiment name", default="regression")
    parser.add_argument('--img_size', type=int, help="image size", default=192)
    parser.add_argument('--batch_size', type=int, help="training batch size", default=2)
    parser.add_argument('--sex', type=int, help="1 for male, 2 for female", default=1)
    parser.add_argument('--gpu_id', type=int, help="number of GPU", default=0)
    parser.add_argument('--epoch', type=int, default=200)
    FLAGS = parser.parse_args()

    pprint(FLAGS)

    exp_name = FLAGS.exp
    img_size = FLAGS.img_size
    batch_size = FLAGS.batch_size
    sex = FLAGS.sex
    gpu_id = FLAGS.gpu_id

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

    # training
    train(n_epoch=FLAGS.epoch, img_size=img_size, sex=sex, batch_size=batch_size)
