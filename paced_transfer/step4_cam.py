import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras import optimizers
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LambdaCallback
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from tqdm import tqdm
import keras

from utils_training import _set_trainable_layers, _pprint
from utils_data import load_sex_ids, load_data, save_obj, load_data_sex
from utils_metric import regression_metric
from config import *


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
    print "[x] compile model on %d GPU(s)" % num_gpu

    model.compile(optimizer=optimizer, loss="mean_squared_error")
    model.summary(print_fn=_pprint)
    return model, base_model


def predict_on_weights(out_base, weights):
    gap = np.average(out_base, axis=(0, 1))
    logit = np.dot(gap, np.squeeze(weights))
    return 1 / (1 + np.e ** (-logit))


def getCAM(image, bone_age, feature_maps, weights, img_size):
    predict = predict_on_weights(feature_maps, weights)
    # Weighted Feature Map
    cam = (predict - 0.5) * np.matmul(feature_maps, weights)
    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    # Resize as image size
    cam_resize = cv2.resize(cam, (img_size, img_size))
    # Format as CV_8UC1 (as applyColorMap required)
    cam_resize = 255 * cam_resize
    cam_resize = cam_resize.astype(np.uint8)
    # Get Heatmap
    heatmap = cv2.applyColorMap(cam_resize, cv2.COLORMAP_JET)
    # Zero out
    heatmap[np.where(cam_resize <= 100)] = 0

    print "image shape = {}, heatmap shape = {}, cam shape = {}".format(image.shape, heatmap.shape, cam_resize.shape)
    print "image dtype = {}, heatmap dtype = {}, cam dtype = {}".format(image.dtype, heatmap.dtype, cam_resize.dtype)
    out = cv2.addWeighted(src1=image, alpha=0.8, src2=heatmap, beta=0.4, gamma=0)
    out = cv2.resize(out, dsize=(400, 400))

    text = 'bone age %.2fm, predict %.2fm' % (bone_age*SCALE, predict*SCALE)
    cv2.putText(out, text, (100, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)
    return out, predict


def train(img_size, start_layer, learning_rate, n_epoch, data_x, data_y):
    model, base_model = _build_regressor(img_size, 1, start_layer, learning_rate)

    weights_history = []
    get_weights_cb = LambdaCallback(on_batch_end=lambda batch,
                                    logs: weights_history.append(model.layers[-1].get_weights()[0]))

    data_gen = ImageDataGenerator(rotation_range=180, zoom_range=0.1, horizontal_flip=True)
    data_gen.fit(data_x)
    history = model.fit_generator(data_gen.flow(data_x, data_y, batch_size=batch_size),
                                               validation_data=(data_x[:1000], data_y[:1000]),
                                               workers=4,
                                               callbacks=[get_weights_cb],
                                               use_multiprocessing=True,
                                               epochs=n_epoch)

    save_obj(weights_history, name=model_out_dir+"/weights.pkl")
    base_model.save(model_out_dir+"/base_model.h5")
    model.save(model_out_dir+"/model.h5")

    return base_model, weights_history


def inference(data_ids, data_y, base_model, weights_history, img_size):
    for i in tqdm(range(len(data_ids))):
        _id = data_ids[i]
        img = cv2.imread(RSNA_SEG_ENHANCE + "/{}_seg.png".format(_id), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        ori_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = np.array(img, dtype=np.float32)
        img = xception_input(img)
        int_image = np.array(ori_img, dtype=np.uint8)
        bone_age = data_y[i]

        out_base = base_model.predict(np.expand_dims(img, axis=0))[0]
        cam, prediction = getCAM(int_image, bone_age, out_base, weights_history[-1], img_size)
        distance = np.abs(prediction - bone_age)*SCALE
        plt.imsave(fname=inference_out_dir+"/{}_{}_cam.png".format(distance, _id), arr=cam)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help="experiment name", default="CAM")
    parser.add_argument('--img_size', type=int, help="image size", default=299)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, help="training batch size", default=16)
    parser.add_argument('--sex', type=int, help="0 for all, 1 for male, 2 for female", default=1)
    parser.add_argument('--start_layer', type=int, help="start_layer", default=0)
    parser.add_argument('--n_epoch', type=int, help="training epochs", default=1)
    FLAGS = parser.parse_args()

    exp_name = FLAGS.exp
    gpu_id = FLAGS.gpu_id
    img_size = FLAGS.img_size
    batch_size = FLAGS.batch_size
    sex = FLAGS.sex
    start_layer = FLAGS.start_layer
    n_epoch = FLAGS.n_epoch

    model_out_dir = "E{}_S{}_IMG_{}_START_{}/model".format(exp_name, sex, img_size, start_layer)
    inference_out_dir = "E{}_S{}_IMG_{}_START_{}/cam".format(exp_name, sex, img_size, start_layer)
    if not os.path.isdir(model_out_dir):
        os.makedirs(model_out_dir)
    if not os.path.isdir(inference_out_dir):
        os.makedirs(inference_out_dir)

    print "[x] building models on GPU {}".format(gpu_id)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    learning_rate = 1E-5

    print "[x] load data ..."
    data_ids = load_sex_ids(sex)
    data_x, data_y = load_data_sex(sex, img_size, xception_input)
    print "[x] loaded data_x {}, data_y {} data".format(data_x.shape, data_y.shape)

    base_model, weights_history = train(img_size, start_layer, learning_rate, n_epoch, data_x, data_y)
    inference(data_ids, data_y, base_model, weights_history, img_size)



