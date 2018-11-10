import numpy as np
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import os

from glob import glob
from tqdm import tqdm
from config import *


MID, CARPAL, THUMB = "middle", "carpal", "thumb"
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

"""
def pad_image(img):
    if img.shape[0] > img.shape[1]:
        pad = (img.shape[0] - img.shape[1]) // 2
        pad_tuple = ((0, 0), (pad, pad))
    else:
        pad = (img.shape[1] - img.shape[0]) // 2
        pad_tuple = ((pad, pad), (0, 0))
    padded = np.pad(img, pad_tuple, mode="constant")
    return padded, pad_tuple
"""


def parser_xml(xml_file):
    dom_tree = ET.parse(xml_file)
    objs = dom_tree.findall("object")
    assert len(objs) == 3
    img_h = float(dom_tree.find("size").find("width").text)
    img_w = float(dom_tree.find("size").find("height").text)

    info = {}
    for obj in objs:
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        # x is col, y is row
        xmin = float(bbox.find("xmin").text)
        xmax = float(bbox.find("xmax").text)
        ymin = float(bbox.find("ymin").text)
        ymax = float(bbox.find("ymax").text)
        x_center = (xmax+xmin)/2
        y_center = (ymax+ymin)/2
        row_ratio = y_center / img_w
        col_ratio = x_center / img_h
        info[name] = (row_ratio, col_ratio)
    return info


def resize_img(img, new_size=(256, 256)):
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def load_test_data(data_image_files_dir, annotation_ids, pattern="_crop_seg", img_size=299):
    data_image_files = glob(data_image_files_dir + "/*{}*".format(pattern))
    data_img = []
    data_ids = []
    for data_image_file in tqdm(data_image_files):
        data_id = data_image_file[data_image_file.rfind("/") + 1: data_image_file.find(pattern)]
        if data_id not in annotation_ids:
            img_ori = cv2.imread(data_image_file, cv2.IMREAD_GRAYSCALE)
            img = resize_img(img_ori, (img_size, img_size))
            img = img / 255.
            img = (img - np.mean(img)) / np.std(img)
            data_img.append(img)
            data_ids.append(data_id)

    data_img = np.expand_dims(np.array(data_img, dtype=np.float32), axis=3)
    return data_img, data_ids


def load_training_data(data_image_files_dir, annotation_dir, pattern="_crop_seg", img_size=299):
    data_annotation_files = glob(annotation_dir + "/*{}*".format(pattern))
    data_img = []
    data_label = []
    annotation_ids = []
    for data_annotation_file in tqdm(data_annotation_files):
        data_id = data_annotation_file[data_annotation_file.rfind("/")+1: data_annotation_file.find(pattern)]
        # print "read file %s" % data_annotation_file
        data_image_file = data_image_files_dir + "/{0}{1}.png".format(data_id, pattern)
        img_ori = cv2.imread(data_image_file, cv2.IMREAD_GRAYSCALE)
        xml_info = parser_xml(data_annotation_file)

        img = resize_img(img_ori, (img_size, img_size))
        label = np.array([xml_info[MID][0], xml_info[MID][1],
                          xml_info[THUMB][0], xml_info[THUMB][1],
                          xml_info[CARPAL][0], xml_info[CARPAL][1]], dtype=np.float32)
        img = img / 255.
        img = (img - np.mean(img)) / np.std(img)

        data_img.append(img)
        data_label.append(label)
        annotation_ids.append(data_id)

    data_img = np.expand_dims(np.array(data_img, dtype=np.float32), axis=3)
    data_label = np.array(data_label, dtype=np.float32)
    return data_img, data_label, annotation_ids


def visual_image(img, coord, visual_dir, output_name, is_train=False, radius=1):
    if is_train:
        plt.imsave(fname=visual_dir + "/{}_ori.png".format(output_name), arr=img, cmap="gray")
        img = cv2.imread(visual_dir + "/{}_ori.png".format(output_name), cv2.IMREAD_COLOR)
    img_w = img.shape[0]
    img_h = img.shape[1]
    middle_finger_y, middle_finger_x = int(coord[0] * img_w), int(coord[1] * img_h)
    cv2.circle(img, (middle_finger_x, middle_finger_y), radius, RED, -1)

    thumb_finger_y, thumb_finger_x = int(coord[2] * img_w), int(coord[3] * img_h)
    cv2.circle(img, (thumb_finger_x, thumb_finger_y), radius, GREEN, -1)

    carpal_y, carpal_x = int(coord[4] * img_w), int(coord[5] * img_h)
    cv2.circle(img, (carpal_x, carpal_y), radius, BLUE, -1)

    plt.imsave(fname=visual_dir + "/{}_annotate.png".format(output_name), arr=img)


def visualize_data(visual_dir="tmp"):
    if not os.path.isdir(visual_dir):
        os.makedirs(visual_dir)
    data_dir = "./interactive_segmentation_dice1.0_pixel1.0_GPU0/segmented_all/"
    data_imgs, data_labels, _ = load_training_data(data_dir, annotation_dir=RSNA_GT_KEY_POINT, pattern="_crop_seg")
    print "load image %d" % data_imgs.shape[0]
    for i in range(data_imgs.shape[0]):
        img = data_imgs[i]
        label = data_labels[i]

        plt.imsave(fname=visual_dir+"/{}_img.png".format(i), arr=np.squeeze(img), cmap="gray")
        img = cv2.imread(visual_dir+"/{}_img.png".format(i), cv2.IMREAD_COLOR)

        visual_image(img, label, visual_dir, output_name=str(i))

if __name__ == "__main__":
    visualize_data()