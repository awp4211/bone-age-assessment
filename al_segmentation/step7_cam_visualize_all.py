import cv2
import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_input

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_input
from keras.applications.xception import preprocess_input as xception_input


from utils_data import load_obj
from step4_regression_with_cam import _build_regresser
from step6_cam import getCAM
from config import RSNA_TRAIN_CSV, RSNA_SEG_ENHANCE

from glob import glob
from tqdm import tqdm


def load_img(data_id=None, img_size=256, preprocess_fn=xception_input):
    data_dir = RSNA_SEG_ENHANCE
    img_file_name = data_dir + "/{}_seg.png".format(data_id)
    img = cv2.imread(img_file_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    ori_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.array(img, dtype=np.float32)
    img = preprocess_fn(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    int_image = np.array(ori_img, dtype=np.uint8)
    return img, int_image


def get_ba_by_id(img_id):
    dataset_df = pd.read_csv(RSNA_TRAIN_CSV)
    boneages = dataset_df[dataset_df.id==img_id].boneage
    boneage = boneages[boneages.index[0]]
    return boneage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help="training batch size", default=32)
    parser.add_argument('--model_name', type=str, help="model name: inception_v3 ....", default="inception_v3")
    parser.add_argument('--exp', type=str, help="experiment name", default="cam")
    parser.add_argument('--sex', type=int, help="0 for all, 1 for male, 2 for female", default=1)
    parser.add_argument('--augment', type=str, help="augment data", default="false")
    parser.add_argument('--fine_tune', type=str, help="fine tune pretrained layer", default="3")
    parser.add_argument('--num_gpu', type=int, default=1)
    FLAGS = parser.parse_args()

    num_gpu = FLAGS.num_gpu
    batch_size = FLAGS.batch_size * num_gpu
    model_name = FLAGS.model_name
    exp_name = FLAGS.exp
    sex = FLAGS.sex
    augment_data = True if FLAGS.augment == "true" else False
    fine_tune = FLAGS.fine_tune

    metric_out_dir = "E{}_M{}_S{}_A{}_F{}/metric".format(exp_name, model_name, sex, augment_data, fine_tune)
    model_out_dir = "E{}_M{}_S{}_A{}_F{}/model".format(exp_name, model_name, sex, augment_data, fine_tune)
    cam_out_dir = "E{}_M{}_S{}_A{}_F{}/cam_visualize_all".format(exp_name, model_name, sex, augment_data, fine_tune)
    cam_out_dir2 = "E{}_M{}_S{}_A{}_F{}/cam_visualize_all2".format(exp_name, model_name, sex, augment_data, fine_tune)

    if not os.path.isdir(cam_out_dir):
        os.makedirs(cam_out_dir)
    if not os.path.isdir(cam_out_dir2):
        os.makedirs(cam_out_dir2)

    if model_name == "inception_v3":
        preprocess_fn = inception_v3_input
    elif model_name == "inception_resnet_v2":
        preprocess_fn = inception_resnet_input
    elif model_name == "xception":
        preprocess_fn = xception_input
    else:
        raise ValueError("Not a supported model name")

    print "[x] load saved model file"
    weights_history = load_obj(model_out_dir + "/weights_history.pkl")
    model, input_shape, base_model = _build_regresser(model_name, weights="imagenet", num_gpu=1, fine_tune=fine_tune)
    base_model.load_weights(filepath=model_out_dir + "/base_model.h5")

    imgs = glob(RSNA_SEG_ENHANCE+"/*.*")
    for img in tqdm(imgs):
        img_id = int(img[img.rfind("/")+1:img.rfind("_")])
        bone_age = get_ba_by_id(img_id)
        img, int_img = load_img(img_id, img_size=299, preprocess_fn=preprocess_fn)
        out = getCAM(image=int_img,
                     feature_maps=base_model.predict(np.expand_dims(img, axis=0))[0],
                     weights=weights_history[-1])

        cv2.imwrite(cam_out_dir+"/{}_{}.png".format(bone_age, img_id), out)
        cv2.imwrite(cam_out_dir2+"/{}.png".format(img_id), out)

