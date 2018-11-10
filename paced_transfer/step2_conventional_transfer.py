import numpy as np
import argparse
import os
import pprint
import keras
import time
import threading
import tensorflow as tf

from keras.utils import multi_gpu_model
from keras import optimizers
from keras.backend.tensorflow_backend import set_session

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import *
from utils_training import _set_trainable_layers, _pprint
from utils_data import load_sex_ids, load_data, save_obj, load_data_sex
from utils_metric import regression_metric


def _build_regressor(img_size=299, num_gpu=1, start_layer=-1, learning_rate=1E-4):
    input_shape = (img_size, img_size, 3)
    base_model = Xception(input_shape=input_shape, weights="imagenet", include_top=False)
    if start_layer == -1:
        _set_trainable_layers(base_model, len(base_model.layers))
    else:
        _set_trainable_layers(base_model, start_layer)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation=keras.activations.relu)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = optimizers.RMSprop(lr=learning_rate, decay=0.95)
    if num_gpu > 1:
        model = multi_gpu_model(model, num_gpu)
    print "[x] compile model on %d GPU(s)" % num_gpu

    model.compile(optimizer=optimizer, loss="mean_squared_error")
    model.summary(print_fn=_pprint)
    return model


class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get("loss"))


def train(n_epoch=N_TRAINING_EPOCH, img_size=299, sex=0, batch_size=16, num_gpu=1, start_layer=-1):
    assert start_layer in [-1, XCEPTION_EXIT_START, XCEPTION_MID_START, XCEPTION_ENTRY_START, 0]
    assert sex in [0, 1, 2]

    # learning rate
    if start_layer == -1:
        learning_rate = 1E-3
    elif start_layer == XCEPTION_EXIT_START:
        learning_rate = 1E-4
    elif start_layer == XCEPTION_MID_START:
        learning_rate = 5E-5
    elif start_layer == XCEPTION_ENTRY_START:
        learning_rate = 1E-5
    else:
        learning_rate = 5E-6

    model = _build_regressor(img_size, num_gpu, start_layer, learning_rate)

    print "[x] load data ..."
    data_x, data_y = load_data_sex(sex, img_size, xception_input)
    print "[x] loaded data_x {}, data_y {} data".format(data_x.shape, data_y.shape)

    data_gen = ImageDataGenerator(
        rotation_range=180,
        zoom_range=0.1,
        horizontal_flip=True
    )

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y)
    data_gen.fit(x_train)

    model_callback = ModelCheckpoint(filepath=model_out_dir+"/model.h5", verbose=1, save_best_only=True)
    loss_callback = LossHistory()
    model.fit_generator(data_gen.flow(x_train, y_train, batch_size=batch_size),
                                           validation_data=(x_test, y_test),
                                           workers=4,
                                           callbacks=[model_callback, loss_callback],
                                           use_multiprocessing=True,
                                           epochs=n_epoch)
    save_obj(obj=loss_callback.losses, name=metric_out_dir+"/losses_S{}.pkl".format(start_layer))

    y_true = []
    y_pred = []
    for mini_batch in range(int(data_x.shape[0] // batch_size)):
        pred_y = model.predict_on_batch(data_x[mini_batch*batch_size:(mini_batch+1)*batch_size])
        for i in range(batch_size):
            y_true.append(data_y[mini_batch*batch_size+i] * SCALE)
            y_pred.append(pred_y[i] * SCALE)

    evs, mae, mse, meae, r2s, ccc = regression_metric(np.array(y_true), np.array(y_pred))
    save_obj({"evs": evs, "mae": mae, "mse": mse, "meae": meae, "r2s": r2s, "ccc": ccc},
             name=metric_out_dir + "/evaluate_S{}.pkl".format(start_layer))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help="experiment name", default="rm")
    parser.add_argument('--img_size', type=int, help="image size", default=128)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, help="training batch size", default=16)
    parser.add_argument('--sex', type=int, help="0 for all, 1 for male, 2 for female", default=1)
    parser.add_argument('--start_layer', type=int, help="start_layer", default=-1)
    parser.add_argument('--n_epoch', type=int, help="training epochs", default=100)
    FLAGS = parser.parse_args()

    pprint.pprint(FLAGS)

    exp_name = FLAGS.exp
    gpu_id = FLAGS.gpu_id
    img_size = FLAGS.img_size
    batch_size = FLAGS.batch_size
    sex = FLAGS.sex
    start_layer = FLAGS.start_layer
    n_epoch = FLAGS.n_epoch

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

    # training
    train(n_epoch=n_epoch,
          img_size=img_size,
          sex=sex,
          batch_size=batch_size,
          num_gpu=1,
          start_layer=start_layer)
