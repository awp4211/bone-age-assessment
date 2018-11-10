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

from keras.utils import multi_gpu_model
from keras import optimizers
from keras.backend.tensorflow_backend import set_session

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_input

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_input

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_input

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, explained_variance_score
from tqdm import tqdm

from config import *
from utils_data import save_obj, load_obj


def regression_metric(y_true, y_pred):
    # explained_variance_score
    evs = explained_variance_score(y_true, y_pred)
    # mean_absolute_error
    mae = mean_absolute_error(y_true, y_pred)
    # mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    # median_absolute_error
    meae = median_absolute_error(y_true, y_pred)
    # r^2_score
    r2s = r2_score(y_true, y_pred)
    ccc = _ccc(y_true, y_pred)
    return evs, mae, mse, meae, r2s, ccc


def _ccc(y_true, y_pred):
    x_mean = np.average(y_true)
    y_mean = np.average(y_pred)
    n = y_true.shape[0]
    s_xy = np.sum(np.multiply(y_true-x_mean, y_pred-y_mean)) / n
    s_x2 = np.sum([np.power(e, 2) for e in (y_true - x_mean)]) / n
    s_y2 = np.sum([np.power(e, 2) for e in (y_pred - y_mean)]) / n
    return 2*s_xy / (s_x2+s_y2+np.power(x_mean-y_mean, 2))


def load_data(sex=0, img_size=256, augment=False, preprocess_fn=inception_v3_input, debug=False):
    """
    :param sex: 0 for all, 1 for male, 2 for female
    :return:
    """
    data_dir = RSNA_SEG_ENHANCE
    dataset_df = pd.read_csv(RSNA_TRAIN_CSV)
    if sex == 0:
        boneages = dataset_df.boneage
        ids = dataset_df.id
    elif sex == 1:
        boneages = dataset_df[dataset_df.male==True].boneage
        ids = dataset_df[dataset_df.male==True].id
    elif sex == 2:
        boneages = dataset_df[dataset_df.male==False].boneage
        ids = dataset_df[dataset_df.male==False].id

    bas_l = []
    ids_l = []
    if debug:
        ids = ids[:100]

    n_data = len(ids)
    for i in range(len(ids)):
        boneage = boneages.ix[boneages.index[i]]
        ids_l.append(ids.ix[ids.index[i]])
        bas_l.append(boneage)

    x = []
    y = []
    for i in tqdm(range(n_data)):
        id = ids_l[i]
        boneage = bas_l[i]
        img_file_name = data_dir + "/{}_seg.png".format(id)
        if os.path.isfile(img_file_name):
            img = cv2.imread(img_file_name, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            img = np.array(img, dtype=np.float32)
            img = preprocess_fn(img)
            x.append(img)
            y.append(boneage)

            if augment:
                flipped = cv2.flip(img, 1) # horzational flip
                for angle in range(-60, 61, 30):
                    M = cv2.getRotationMatrix2D(center=(img.shape[0]/2, img.shape[1]/2), angle=angle, scale=1)
                    dst_ori = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))
                    dst_flip = cv2.warpAffine(flipped, M, (img.shape[0], img.shape[1]))
                    x.append(dst_ori)
                    x.append(dst_flip)

                    y.append(boneage)
                    y.append(boneage)

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    y = y/SCALE
    print "[x] load %d data" % x.shape[0]
    return x, y


def _build_regresser(model_name, weights, num_gpu, fine_tune):
    input_shape = (192, 192, 3)
    if model_name == "inception_v3":
        base_model = InceptionV3(input_shape=input_shape, weights=weights, include_top=False)
    elif model_name == "inception_resnet_v2":
        base_model = InceptionResNetV2(input_shape=input_shape, weights=weights, include_top=False)
    elif model_name == "xception":
        base_model = Xception(input_shape=input_shape, weights=weights, include_top=False)
    else:
        raise ValueError("NOT A SUPPORT MODEL")

    if model_name == "inception_v3":
        if fine_tune == INCEPTION_V3_INCEPTION_3:
            start = INCEPTION_V3_INCEPTION_3_START
        elif fine_tune == INCEPTION_V3_INCEPTION_4:
            start = INCEPTION_V3_INCEPTION_4_START
        elif fine_tune == INCEPTION_V3_INCEPTION_5:
            start = INCEPTION_V3_INCEPTION_5_START
        elif fine_tune == FINE_TUNE_ALL:
            start = -1
    elif model_name == "inception_resnet_v2":
        if fine_tune == INCEPTION_RESNET_V2_INCEPTION_A:
            start = INCEPTION_RESNET_V2_INCEPTION_A_START
        elif fine_tune == INCEPTION_RESNET_V2_INCEPTION_B:
            start = INCEPTION_RESNET_V2_INCEPTION_B_START
        elif fine_tune == INCEPTION_RESNET_V2_INCEPTION_C:
            start = INCEPTION_RESNET_V2_INCEPTION_C_START
        elif fine_tune == FINE_TUNE_ALL:
            start = -1
    elif model_name == "xception":
        if fine_tune == XCEPTION_ENTRY:
            start = XCEPTION_ENTRY_START
        elif fine_tune == XCEPTION_MID:
            start = XCEPTION_MID_START
        elif fine_tune == XCEPTION_EXIT:
            start = XCEPTION_EXIT_START
        elif fine_tune == FINE_TUNE_ALL:
            start = -1
    else:
        raise ValueError("NOT A SUPPORT MODEL")

    for i, layer in enumerate(base_model.layers):
        if i < start:
            layer.trainable = False
        else:
            layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation=keras.activations.relu)(x)
    print predictions.get_shape()
    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = optimizers.RMSprop(lr=0.005, decay=0.95)
    if num_gpu > 1:
        model = multi_gpu_model(model, num_gpu)
    print "[x] compile model on %d GPU(s)" % num_gpu
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    return model, input_shape, base_model


def _batch_cvt(batch_data):
    imgs = []
    for data in batch_data:
        img = np.squeeze(data)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imgs.append(img)
    imgs = np.array(imgs, dtype=np.float32)
    return imgs


def train(model_name,
          weights="imagenet",
          n_epoch=N_TRAINING_EPOCH,
          sex=0,
          batch_size=16,
          augment=False,
          num_gpu=1,
          fine_tune=""):
    model, input_shape, base_model = _build_regresser(model_name, weights, num_gpu, fine_tune)
    if model_name == "inception_v3":
        preprocess_fn = inception_v3_input
    elif model_name == "inception_resnet_v2":
        preprocess_fn = inception_resnet_input
    elif model_name == "xception":
        preprocess_fn = xception_input
    else:
        raise ValueError("Not a supported model name")

    data_x, data_y = load_data(sex, input_shape[0], augment, preprocess_fn)
    print "[x] total data size {} G".format(sys.getsizeof(data_x)/1024**3)
    best_loss = np.inf
    for epoch in tqdm(range(n_epoch)):
        print "========================================================================================================"
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0, shuffle=True)
        for mini_batch in range(x_train.shape[0] // batch_size):
            batch_x = x_train[mini_batch * batch_size: (mini_batch + 1) * batch_size]
            batch_x_cvt = _batch_cvt(batch_x)
            batch_y = y_train[mini_batch * batch_size: (mini_batch + 1) * batch_size]
            loss = model.train_on_batch(x=batch_x_cvt, y=batch_y)
            if mini_batch % 100 == 0:
                print "--epoch {}, mini_batch {}, loss {}".format(epoch, mini_batch, loss)

        # test
        print "[x] test in epoch {}".format(epoch)
        losses = 0.0
        for mini_batch in range(x_test.shape[0] // batch_size):
            batch_x = x_test[mini_batch * (batch_size): (mini_batch + 1) * batch_size]
            batch_y = y_test[mini_batch * (batch_size): (mini_batch + 1) * batch_size]
            batch_x_cvt = _batch_cvt(batch_x)
            loss = model.test_on_batch(batch_x_cvt, batch_y)
            losses += loss
        losses = losses/(x_test.shape[0] // batch_size)
        if losses < best_loss:
            best_loss = losses
            model.save_weights(model_out_dir + "/epoch_{}.h5".format(epoch))
        print "== epoch {}, test loss {}".format(epoch, losses)

        # test and metric
        print "[x] predict in epoch {}".format(epoch)
        y_true = []
        y_pred = []
        for mini_batch in range(x_test.shape[0] // batch_size):
            batch_x = x_test[mini_batch*(batch_size): (mini_batch+1)*batch_size]
            batch_y = y_test[mini_batch*(batch_size): (mini_batch+1)*batch_size]
            batch_x_cvt = _batch_cvt(batch_x)
            pred_y = model.predict_on_batch(batch_x_cvt)
            for i in range(batch_size):
                y_true.append(batch_y[i]*SCALE)
                y_pred.append(pred_y[i]*SCALE)

        evs, mae, mse, meae, r2s, ccc = regression_metric(np.array(y_true), np.array(y_pred))
        save_obj({"evs": evs,
                  "mae": mae,
                  "mse": mse,
                  "meae": meae,
                  "r2s": r2s,
                  "ccc": ccc,
                  "loss": losses},
                  name=metric_out_dir+"/epoch_{}.pkl".format(epoch))
        print "[x] epoch {}, evs {}, mae {}, mse {}, meae {}, r2s {}, ccc {}".format(epoch, evs, mae, mse, meae, r2s, ccc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help="training batch size", default=16)
    parser.add_argument('--model_name', type=str, help="model name: inception_v3 ....", default="xception")
    parser.add_argument('--exp', type=str, help="experiment name", default="regression")
    parser.add_argument('--sex', type=int, help="0 for all, 1 for male, 2 for female", default=1)
    parser.add_argument('--augment', type=str, help="augment data", default="false")
    parser.add_argument('--fine_tune', type=str, help="fine tune pretrained layer", default="0")
    parser.add_argument('--num_gpu', type=int, default=1)
    FLAGS = parser.parse_args()

    pprint.pprint(FLAGS)

    num_gpu = FLAGS.num_gpu
    batch_size = FLAGS.batch_size * num_gpu
    model_name = FLAGS.model_name
    exp_name = FLAGS.exp
    sex = FLAGS.sex
    augment_data = True if FLAGS.augment == "true" else False
    fine_tune = FLAGS.fine_tune

    print "[x] building models on {} GPUs".format(num_gpu)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    metric_out_dir = "E{}_M{}_S{}_A{}_F{}/metric".format(exp_name, model_name, sex, augment_data, fine_tune)
    model_out_dir = "E{}_M{}_S{}_A{}_F{}/model".format(exp_name, model_name, sex, augment_data, fine_tune)
    if not os.path.isdir(metric_out_dir):
        os.makedirs(metric_out_dir)
    if not os.path.isdir(model_out_dir):
        os.makedirs(model_out_dir)

    # training
    train(model_name=model_name, weights="imagenet", n_epoch=N_TRAINING_EPOCH, sex=sex,
          batch_size=batch_size, augment=augment_data, num_gpu=num_gpu, fine_tune=fine_tune)

    """
    # save weight and base model
    save_obj(weights_history, name=model_out_dir + "/weights_history.pkl")
    base_model.save_weights(filepath=model_out_dir + "/base_model.h5")
    """
