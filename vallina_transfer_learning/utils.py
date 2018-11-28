import numpy as np
import pandas as pd
import cv2
import os
import pickle
import h5py
import keras

from keras.utils import multi_gpu_model
from keras import optimizers
from keras.backend.tensorflow_backend import set_session

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_input

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_input

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_input

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, explained_variance_score
from tqdm import tqdm
from config import *


def _batch_cvt(batch_data):
    imgs = []
    for data in batch_data:
        img = np.squeeze(data)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imgs.append(img)
    imgs = np.array(imgs, dtype=np.float32)
    return imgs


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def load_hdf5(infile, key="data"):
    with h5py.File(infile, "r") as f:
        return f[key][()]


def write_hdf5(value, outfile, key="data"):
    with h5py.File(outfile, "w") as f:
        f.create_dataset(key, data=value, dtype=value.dtype)


def load_sex_ids(sex=0):
    dataset_df = pd.read_csv(RSNA_TRAIN_CSV)
    ids = []
    if sex == 0:
        for i in range(len(dataset_df.index)):
            id = dataset_df.ix[dataset_df.index[i]]['id']
            ids.append(id)
    elif sex == 1:
        sex_df = dataset_df[dataset_df.male == True]
        for i in range(len(sex_df.index)):
            id = dataset_df.ix[sex_df.index[i]]['id']
            ids.append(id)
    elif sex == 2:
        sex_df = dataset_df[dataset_df.male == False]
        for i in range(len(sex_df.index)):
            id = dataset_df.ix[sex_df.index[i]]['id']
            ids.append(id)

    return np.array(ids)


def load_data(sex=0, img_size=299, batch_size=32, augment_times=5):
    data_ids = load_sex_ids(sex)
    select_ids = np.random.permutation(data_ids)[:batch_size]
    batch_x, batch_y = augment_data_with_ids(select_ids, img_size, augment_times=augment_times)
    return _batch_cvt(batch_x), batch_y


def augment_data_with_ids(ids, img_size=256, preprocess_fn=xception_input, data_dir=RSNA_TRAIN_DATA, augment_times=5):
    imgs = []
    boneages = []
    for id in ids:
        ims, bas = _augment_data_with_id(id, img_size, preprocess_fn, data_dir, augment_times)
        imgs.extend(ims)
        boneages.extend(bas)

    imgs = np.array(imgs, dtype=np.float32)
    boneages = np.array(boneages, dtype=np.float32)
    indexes = np.random.permutation(range(imgs.shape[0]))
    imgs = imgs[indexes]
    boneages = boneages[indexes]
    return imgs, boneages


def load_data_sex(sex, img_size=256, preprocess_fn=xception_input, data_dir=RSNA_TRAIN_DATA):
    ids = load_sex_ids(sex)
    imgs = []
    boneages = []
    for _id in tqdm(ids):
        img, boneage = load_single_with_id(_id, img_size, preprocess_fn, data_dir)
        imgs.append(img)
        boneages.append(boneage)
    return np.array(imgs, dtype=np.float32), np.array(boneages, dtype=np.float32)


def load_single_with_id(_id, img_size=256,  preprocess_fn=xception_input, data_dir=RSNA_TRAIN_DATA):
    dataset_df = pd.read_csv(RSNA_TRAIN_CSV)
    boneage = dataset_df.loc[dataset_df["id"] == _id].boneage.values[0] / SCALE
    img_file_name = data_dir + "/{}.png".format(_id)
    img = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size))
    img = preprocess_fn(np.array(img, dtype=np.float32))
    return img, boneage


def _pprint(content):
    if content.startswith("T") or content.startswith("N"):
        print content


class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get("loss"))


def _augment_data_with_id(_id, img_size=256, preprocess_fn=xception_input, data_dir=RSNA_TRAIN_DATA, augment_times=5):
    """
    Online sampling with data augmentation
    :param _id:
    :param img_size:
    :param preprocess_fn:
    :param dataset_df:
    :param data_dir:
    :param augment_times:
    :return:
    """
    imgs = []
    boneages = []
    dataset_df = pd.read_csv(RSNA_TRAIN_CSV)

    boneage = dataset_df.loc[dataset_df["id"]==_id].boneage.values[0] / SCALE
    img_file_name = data_dir + "/{}.png".format(_id)
    img = cv2.imread(img_file_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))

    imgs.append(preprocess_fn(np.array(img, dtype=np.float32)))
    boneages.append(boneage)

    if augment_times > 0:
        flipped = cv2.flip(img, 1)  # horzational flip
        imgs.append(preprocess_fn(np.array(flipped, dtype=np.float32)))
        boneages.append(boneage)
        for i in range(augment_times):
            angle = np.random.randint(0, 360)
            M = cv2.getRotationMatrix2D(center=(img.shape[0] / 2, img.shape[1] / 2), angle=angle, scale=1)
            dst_ori = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            dst_flip = cv2.warpAffine(flipped, M, (img.shape[0], img.shape[1]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            imgs.append(preprocess_fn(np.array(dst_ori, dtype=np.float32)))
            imgs.append(preprocess_fn(np.array(dst_flip, dtype=np.float32)))

            boneages.append(boneage)
            boneages.append(boneage)

    return imgs, boneages


if __name__ == "__main__":
    """
    data_x, data_y = \
        load_data(sex=0, img_size=256, augment=False, preprocess_fn=inception_v3_input, debug=True, regression=False)
    """
    import matplotlib.pyplot as plt

    imgs, boneages = load_data_sex(1)

    print imgs.shape
    print boneages.shape