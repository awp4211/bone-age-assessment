import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import cv2

from tqdm import tqdm
from glob import glob

import utils_data

from tf_model_unet import unet
from tf_ops import dice_coef_loss, pixelwise_cross_entropy
from config import *
from utils_metrics import threshold_by_otsu

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help="training batch size", default=4)
parser.add_argument('--dice', type=float, help="dice coefficient loss ratio", default=1.0)
parser.add_argument('--pixel', type=float, help="pixelwise cross entropy loss ratio", default=1.0)
parser.add_argument('--gpu', type=int, help="GPU number", default=0)
parser.add_argument('--exp', type=str, help="experiment name", default="interactive_segmentation")
FLAGS = parser.parse_args()

batch_size = FLAGS.batch_size
dice_ratio = FLAGS.dice
pixel_ratio = FLAGS.pixel
gpu = FLAGS.gpu

experiment_name = "{}_dice{}_pixel{}_GPU{}".format(FLAGS.exp, dice_ratio, pixel_ratio, gpu)
img_out_dir = "{}/segmentation_results".format(experiment_name)
model_out_dir = "{}/model".format(experiment_name)
metrics_out_dir = "{}/metrics".format(experiment_name)
inference_dir = "{}/inferenced_result".format(experiment_name)
segmented_all = "{}/segmented_all".format(experiment_name)
if not os.path.isdir(inference_dir):
    os.makedirs(inference_dir)
if not os.path.isdir(segmented_all):
    os.makedirs(segmented_all)


print "[x] building models on GPU: {}".format(gpu)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# create model 0
is_train = tf.placeholder(tf.bool)
print "[x] building Model 0"
tf.set_random_seed(1377)
x_0 = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], name="x_0")
y_0 = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], name="y_0")
y_prob0, conv9_feature0 = unet(x_0, is_train, n_filters=32, name="u-net_0")
dice_coef_loss_0 = dice_coef_loss(y_prob0, y_0)
pixelwise_cross_entropy_0 = pixelwise_cross_entropy(y_prob0, y_0)
loss_0 = dice_ratio * dice_coef_loss_0 + pixel_ratio * pixelwise_cross_entropy_0

print "[x] building Model 1"
tf.set_random_seed(7988)
x_1 = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], name="x_1")
y_1 = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], name="y_1")
y_prob1, conv9_feature1 = unet(x_1, is_train, n_filters=32, name="u-net_1")
dice_coef_loss_1 = dice_coef_loss(y_prob1, y_1)
pixelwise_cross_entropy_1 = pixelwise_cross_entropy(y_prob1, y_1)
loss_1 = dice_ratio * dice_coef_loss_1 + pixel_ratio * pixelwise_cross_entropy_1


print "[x] restore model ..."
saver = tf.train.Saver()
model_file = tf.train.latest_checkpoint(model_out_dir)
saver.restore(sess, model_file)

print "[x] loading data ..."
annotated_ids = utils_data.load_annotated_data_np(data_dir=RSNA_GT_NP_ANNOTATED, return_ids=True)
print "[x] load annotated data ..."
annotated_x, annotated_y, annotated_ids = utils_data.load_annotated_data_np(data_dir=RSNA_GT_NP_ANNOTATED, return_ids=False)
print "[x] load unannotated data ..."
unannotated_x, unannotated_ids = utils_data.load_unannotated_data_np(data_dir=RSNA_GT_NP_UNANNOTATED, annotated_ids=annotated_ids)
print "[x] loaded annotated data: {}, unannotated data: {}".format(annotated_x.shape[0], unannotated_x.shape[0])

# Segment unannotated data
def segment_unannotated_image(sess, input_tensor, output_tensor, is_training, image_data):
    images = []
    for k in range(batch_size):
        images.append(image_data)
    images = np.array(images, dtype=np.float32)
    [segmentation] = sess.run([output_tensor], feed_dict={input_tensor: images, is_training: False})
    return np.squeeze(segmentation[0], axis=2)


def resize_to_ori(image_data, image_data_id, data_dir=RSNA_DATA_DIR):
    img_ori = cv2.imread(data_dir+"/{}.png".format(image_data_id), cv2.IMREAD_GRAYSCALE)
    if img_ori.shape[0] > img_ori.shape[1]:
        pad = (img_ori.shape[0] - img_ori.shape[1]) // 2
        pad_tuple = ((0, 0), (pad, pad))
    else:
        pad = (img_ori.shape[1] - img_ori.shape[0]) // 2
        pad_tuple = ((pad, pad), (0, 0))
    padded = np.pad(img_ori, pad_tuple, mode="constant")
    resized_img = cv2.resize(image_data, (padded.shape[1], padded.shape[0]), interpolation=cv2.INTER_AREA)
    return resized_img, padded


def crop_hand_bin(img_bin):
    try:
        smooth_w, smooth_h = img_bin.shape[0]//30, img_bin.shape[1]//30
        min_row = np.min(np.where(img_bin==1)[0])
        max_row = np.max(np.where(img_bin==1)[0])
        min_col = np.min(np.where(img_bin==1)[1])
        max_col = np.max(np.where(img_bin==1)[1])
        min_row = np.clip(min_row-smooth_w, 0, min_row)
        max_row = np.clip(max_row+smooth_w, max_row, img_bin.shape[0]-1)
        min_col = np.clip(min_col-smooth_h, 0, min_col)
        max_col = np.clip(max_col+smooth_h, max_col, img_bin.shape[1]-1)
        return min_row, max_row, min_col, max_col
    except:
        print "expection happened"
        return 0, img_bin.shape[0], 0, img_bin.shape[1]


def pad_img_2_rect(img):
    if img.shape[0] > img.shape[1]:
        pad = (img.shape[0] - img.shape[1]) // 2
        pad_tuple = ((0, 0), (pad, pad))
    else:
        pad = (img.shape[1] - img.shape[0]) // 2
        pad_tuple = ((pad, pad), (0, 0))
    padded = np.pad(img, pad_tuple, mode="constant")
    return padded

# inference with model 0
for i in range(unannotated_x.shape[0]):
    image_id = unannotated_ids[i]
    print "inference {}".format(image_id)
    segmentated_data_0 = segment_unannotated_image(sess, x_0, y_prob0, is_train, unannotated_x[i])
    resized_segmentated_data_0, padded_ori_img = resize_to_ori(segmentated_data_0, image_id, data_dir=RSNA_DATA_DIR)
    img_seg_0 = np.multiply(resized_segmentated_data_0, padded_ori_img)
    plt.imsave(fname=inference_dir+"/{}_x.png".format(image_id), arr=np.squeeze(unannotated_x[i], axis=2), cmap="gray")
    plt.imsave(fname=inference_dir+"/{}_y.png".format(image_id), arr=segmentated_data_0, cmap="gray")

    plt.imsave(fname=segmented_all+"/{}_ori.png".format(image_id), arr=padded_ori_img, cmap="gray")
    plt.imsave(fname=segmented_all+"/{}_bin.png".format(image_id), arr=resized_segmentated_data_0, cmap="gray")
    plt.imsave(fname=segmented_all+"/{}_seg.png".format(image_id), arr=img_seg_0, cmap="gray")

    r_min, r_max, c_min, c_max = crop_hand_bin(np.asarray(resized_segmentated_data_0, dtype=np.int32))
    plt.imsave(fname=segmented_all+"/{}_crop_ori.png".format(image_id),
               arr=padded_ori_img[r_min: r_max, c_min: c_max], cmap="gray")
    plt.imsave(fname=segmented_all + "/{}_crop_bin.png".format(image_id),
               arr=resized_segmentated_data_0[r_min: r_max, c_min: c_max], cmap="gray")
    plt.imsave(fname=segmented_all + "/{}_crop_seg.png".format(image_id),
               arr=img_seg_0[r_min: r_max, c_min: c_max], cmap="gray")


# Segment annotated data
print "segment annotated data"
annotated_x_files = glob(RSNA_GT+"/*_ori.png")
annotated_y_files = glob(RSNA_GT+"/*_bin.png")
for annotated_x_file in annotated_x_files:
    image_id = annotated_x_file[annotated_x_file.rfind("/")+1: annotated_x_file.find("_")]
    print "segment {}".format(image_id)
    img_x = cv2.imread(annotated_x_file, cv2.IMREAD_GRAYSCALE)
    img_x = pad_img_2_rect(img_x)
    annotated_y_file = RSNA_GT + "/{}_bin.png".format(image_id)
    img_y = cv2.imread(annotated_y_file, cv2.IMREAD_GRAYSCALE)
    img_y = pad_img_2_rect(img_y)
    img_y = img_y / 255.
    img_seg = np.multiply(img_x, img_y)
    plt.imsave(fname=segmented_all+"/{}_ori.png".format(image_id), arr=img_x, cmap="gray")
    plt.imsave(fname=segmented_all+"/{}_bin.png".format(image_id), arr=img_y, cmap="gray")
    plt.imsave(fname=segmented_all+"/{}_seg.png".format(image_id), arr=img_seg, cmap="gray")

    r_min, r_max, c_min, c_max = crop_hand_bin(np.asarray(img_y, dtype=np.int32))
    plt.imsave(fname=segmented_all + "/{}_crop_ori.png".format(image_id),
               arr=img_x[r_min: r_max, c_min: c_max], cmap="gray")
    plt.imsave(fname=segmented_all + "/{}_crop_bin.png".format(image_id),
               arr=img_y[r_min: r_max, c_min: c_max], cmap="gray")
    plt.imsave(fname=segmented_all + "/{}_crop_seg.png".format(image_id),
               arr=img_seg[r_min: r_max, c_min: c_max], cmap="gray")

# Segment new annotated data
print "segment new annotated data"
annotated_y_files = glob(RSNA_GT_NEW+"/*_bin.png")
for annotated_y_file in tqdm(annotated_y_files):
    image_id = annotated_y_file[annotated_y_file.rfind("/")+1: annotated_y_file.find("_")]
    print "segment {}".format(image_id)
    img_y = cv2.imread(annotated_y_file, cv2.IMREAD_GRAYSCALE)
    img_y = pad_img_2_rect(img_y)
    img_y[img_y > 0] = 255
    img_y = img_y / 255.
    annotated_x_file = RSNA_GT_NEW + "/{}.png".format(image_id)
    img_x = cv2.imread(annotated_x_file, cv2.IMREAD_GRAYSCALE)
    img_x = pad_img_2_rect(img_x)
    img_seg = np.multiply(img_x, img_y)
    plt.imsave(fname=segmented_all + "/{}_ori.png".format(image_id), arr=img_x, cmap="gray")
    plt.imsave(fname=segmented_all + "/{}_bin.png".format(image_id), arr=img_y, cmap="gray")
    plt.imsave(fname=segmented_all + "/{}_seg.png".format(image_id), arr=img_seg, cmap="gray")

    r_min, r_max, c_min, c_max = crop_hand_bin(np.asarray(img_y, dtype=np.int32))
    plt.imsave(fname=segmented_all + "/{}_crop_ori.png".format(image_id),
               arr=img_x[r_min: r_max, c_min: c_max], cmap="gray")
    plt.imsave(fname=segmented_all + "/{}_crop_bin.png".format(image_id),
               arr=img_y[r_min: r_max, c_min: c_max], cmap="gray")
    plt.imsave(fname=segmented_all + "/{}_crop_seg.png".format(image_id),
               arr=img_seg[r_min: r_max, c_min: c_max], cmap="gray")