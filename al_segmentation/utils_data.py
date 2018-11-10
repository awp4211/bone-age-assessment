import numpy as np
import cv2
import pickle
import os
import h5py

from tqdm import tqdm
from glob import glob
from config import *


def _read_and_resize(files):
    """
    read images and pad images to rect
    :param files:
    :return:
    """
    data = []
    for f in tqdm(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img.shape[0] > img.shape[1]:
            pad = (img.shape[0] - img.shape[1]) // 2
            pad_tuple = ((0, 0), (pad, pad))
        else:
            pad = (img.shape[1] - img.shape[0]) // 2
            pad_tuple = ((pad, pad), (0, 0))
        img = np.pad(img, pad_tuple, mode="constant")
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        data.append(img)
    return data


def load_unannotated_data(data_dir=RSNA_DATA_DIR, annotated_ids=[]):
    data_images = glob(data_dir+"/*.png")
    data_x_files = []
    unannotated_ids = []
    for data_image in data_images:
        data_id = data_image[data_image.rfind("/")+1: data_image.rfind(".")]
        if data_id not in annotated_ids:
            data_x_files.append(data_image)
            unannotated_ids.append(data_id)

    data_x = np.expand_dims(np.asarray(_read_and_resize(data_x_files), dtype=np.float32), axis=3)
    return data_x, unannotated_ids


def load_unannotated_data_np(data_dir=RSNA_GT_NP_UNANNOTATED, annotated_ids=[]):
    image_x_files = glob(data_dir + "/*_x.h5")
    unannotated_ids = []
    data_x = []
    for image_x_file in image_x_files:
        data_id = image_x_file[image_x_file.rfind("/")+1: image_x_file.find("_")]
        if data_id not in annotated_ids:
            image_x = load_hdf5(image_x_file)
            data_x.append(image_x)
            unannotated_ids.append(data_id)

    data_x = np.expand_dims(np.asarray(data_x, dtype=np.float32), axis=3)
    return data_x, unannotated_ids


def load_annotated_data_np(data_dir=RSNA_GT_NP_ANNOTATED, return_ids=False):
    image_x_files = glob(data_dir+"/*_x.h5")
    if return_ids:
        data_ids = []
        for image_x_file in image_x_files:
            data_id = image_x_file[image_x_file.rfind("/")+1: image_x_file.find("_")]
            data_ids.append(data_id)
        return data_ids
    else:
        data_x = []
        data_y = []
        data_ids = []
        for i in range(len(image_x_files)):
            image_x_file = image_x_files[i]
            data_id = image_x_file[image_x_file.rfind("/")+1: image_x_file.find("_")]
            image_y_file = RSNA_GT_NP_ANNOTATED + "/{}_y.h5".format(data_id)
            image_x = load_hdf5(image_x_file)
            image_y = load_hdf5(image_y_file)
            data_x.append(image_x)
            data_y.append(image_y)
            data_ids.append(data_id)

        data_x = np.expand_dims(np.asarray(data_x, dtype=np.float32), axis=3)
        data_y = np.expand_dims(np.asarray(data_y, dtype=np.float32), axis=3)
        return data_x, data_y, data_ids


def load_annotated_data(data_dir=RSNA_SEGMENT_SAVE_DIR, return_ids=False):
    image_files = glob(data_dir+"/*.png")
    data_x_files = []
    data_y_files = []
    gt_ids = []
    for data_file in image_files:
        data_id = data_file[data_file.rfind('/')+1: data_file.find('_')]
        if data_id in gt_ids:
            continue
        else:
            x_image_path = data_dir + "/{}_ori.png".format(data_id)
            data_x_files.append(x_image_path)
            y_image_path = data_dir + "/{}_bin.png".format(data_id)
            data_y_files.append(y_image_path)
            gt_ids.append(data_id)
    if return_ids:
        return gt_ids

    data_x = np.expand_dims(np.asarray(_read_and_resize(data_x_files), dtype=np.float32), axis=3)
    data_y = np.expand_dims(np.asarray(_read_and_resize(data_y_files), dtype=np.float32), axis=3)
    return data_x, data_y


def shuffle_data(data_x, data_y):
    n_data = data_x.shape[0]
    new_indexes = np.random.permutation(n_data)
    return data_x[new_indexes], data_y[new_indexes], new_indexes


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def similarity_cosine(vec1, vec2):
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    return np.dot(vec1, vec2) / (np.sqrt((vec1**2).sum()) * np.sqrt((vec2**2).sum()))


def euler_distance(vec1, vec2):
    return np.sum(np.power(np.subtract(vec1, vec2), 2))


def load_hdf5(infile, key="data"):
    with h5py.File(infile, "r") as f:
        return f[key][()]


def write_hdf5(value, outfile, key="data"):
    with h5py.File(outfile, "w") as f:
        f.create_dataset(key, data=value, dtype=value.dtype)


if __name__ == '__main__':
    #load_annotated_data(data_dir=RSNA_GT, return_ids=True)
    load_unannotated_data(data_dir=RSNA_DATA_DIR, annotated_ids=[])