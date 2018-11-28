
import numpy as np
import pandas as pd
import os
import argparse
import pprint
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from config import *
from utils_data_gen import build_data_generator
from utils import save_obj, _pprint, LossHistory

# import keras
import keras.backend as K
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras.metrics import mean_absolute_error
from keras import optimizers



bone_age_std = 0
bone_age_mean = 0


def ccc(y_true, y_pred):
    y_true = y_true * bone_age_std + bone_age_mean
    y_pred = y_pred * bone_age_std + bone_age_mean

    x_mean = K.mean(y_true, axis=-1)
    y_mean = K.mean(y_pred, axis=-1)
    return 2 * (y_true -x_mean) * (y_pred - y_mean) / \
            (K.pow(y_true - x_mean, 2) + K.pow(y_pred - y_mean, 2) + K.pow(x_mean - y_mean, 2))


def mae_months(in_gt, in_pred):
    return mean_absolute_error(bone_age_std * in_gt, bone_age_std * in_pred)


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

    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=[mae_months, ccc])

    model.summary(print_fn=_pprint)
    return model


def build_callbacks(model_file_path):
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    checkpoint = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto',
                                       epsilon=0.0001, cooldown=5, min_lr=0.000001)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    loss_callback = LossHistory()
    callbacks_list = [checkpoint, early, reduceLROnPlat, loss_callback]
    return callbacks_list


def train(flags, output_path):
    print "[x] loading data"
    train_gen, valid_gen, bone_mean, bone_std, test_X, test_Y = \
        build_data_generator(model_name=flags.model_name, sex=flags.sex, img_path=flags.image_path,
                             num_per_category=flags.n_samples, img_size=flags.image_size, batch_size=flags.batch_size)
    global bone_age_std
    bone_age_std = bone_std
    global bone_age_mean
    bone_age_mean = bone_mean

    print "[x] building model"
    bone_age_model = build_model(flags, model_file=output_path + "/weights.h5")
    print "[x] building callbacks"
    callbacks = build_callbacks(output_path + "/weights.h5")
    print "[x] fit"

    hist = bone_age_model.fit_generator(train_gen,
                                        validation_data=(test_X, test_Y),
                                        epochs=flags.n_epochs,
                                        callbacks=callbacks,
                                        workers=4,
                                        steps_per_epoch=5000)
    save_obj(hist.history, output_path + "/{}_history.pkl".format(flags.fine_tune))


if __name__ == "__main__":
    # python step1_fine_tune.py --batch_size=16 --model_name=inception_v3 --exp=ENH --sex=1 --fine_tune=-1
    #  --gpu_id=0 --image_path=../data/segmented_enhance --image_size=384 --n_epoch=500 --n_samples=500

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help="training batch size", default=4)
    parser.add_argument('--model_name', type=str, help="model name: inception_v3 ....", default="inception_v3")
    parser.add_argument('--exp', type=str, help="experiment name", default="ENH")
    parser.add_argument('--sex', type=int, help="0 for all, 1 for male, 2 for female", default=2)
    parser.add_argument('--fine_tune', type=int, help="fine tune pretrained layer", default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--image_path', type=str, default=RSNA_TRAIN_DATA)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--n_samples', type=int, default=500)
    FLAGS = parser.parse_args()

    pprint.pprint(FLAGS)

    print "[x] building models on GPU {}".format(FLAGS.gpu_id)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(FLAGS.gpu_id)

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    set_session(tf.Session(config=config))

    output_path = "E{}_M{}_S{}_I{}".format(FLAGS.exp, FLAGS.model_name, FLAGS.sex, FLAGS.image_size)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    train(FLAGS, output_path)
