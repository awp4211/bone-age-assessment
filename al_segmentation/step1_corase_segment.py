from glob import glob
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

from PIL import Image
from PIL import ImageEnhance
from skimage import morphology, measure
from skimage.filters import threshold_otsu
from tqdm import tqdm
from multiprocessing import Pool

from config import *


def load_and_pad(data_file):
    img = cv2.imread(data_file, cv2.IMREAD_GRAYSCALE)
    if img.shape[0] > img.shape[1]:
        pad = (img.shape[0] - img.shape[1]) // 2
        pad_tuple = ((0, 0), (pad, pad))
    else:
        pad = (img.shape[1] - img.shape[0]) // 2
        pad_tuple = ((pad, pad), (0, 0))
    padded = np.pad(img, pad_tuple, mode="constant")
    return padded


def image_enhance_seg(img_files, dilation_k=(5, 5), single_dir=False):
    for i in tqdm(range(len(img_files))):
        data_img_file = img_files[i]
        file_name = data_img_file[data_img_file.rfind("/")+1:data_img_file.rfind(".")]
        print "[x] processing image file %s " % file_name

        img = Image.open(data_img_file)
        img = img.convert('L')
        ori_img = img.convert('L')

        enh_bri = ImageEnhance.Brightness(img)
        brightness = 5
        img = enh_bri.enhance(brightness)
        img = np.array(img)

        # OTSU
        thresh = threshold_otsu(img)
        binary = img > thresh * THRESH_RATIO
        binary = morphology.dilation(binary, selem=np.ones(dilation_k))

        labels = measure.label(binary)
        regions = measure.regionprops(labels)
        labels = [(r.area, r.label) for r in regions]
        if len(labels) > 1:
            labels.sort(reverse=True)
            max_area = labels[1][0]
            for r in regions:
                if r.area <= max_area:
                    for c in r.coords:
                        binary[c[0], c[1]] = False

        binary = morphology.dilation(binary, selem=np.ones(dilation_k))

        if single_dir:
            save_dir = RSNA_SEGMENT_SAVE_DIR + "{}/".format(file_name)
        else:
            save_dir = RSNA_SEGMENT_SAVE_DIR
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # cropped img
        plt.imsave(fname=save_dir + file_name + "_ori.png", arr=np.array(ori_img), cmap="gray")
        padded_ori_img = load_and_pad(save_dir + file_name + "_ori.png")
        plt.imsave(fname=save_dir + file_name + "_pad_ori.png", arr=padded_ori_img, cmap="gray")

        plt.imsave(fname=save_dir + file_name + "_bin.png", arr=binary, cmap=plt.cm.bone)
        padded_bin_img = load_and_pad(save_dir + file_name + "_bin.png")
        plt.imsave(fname=save_dir + file_name + "_pad_bin.png", arr=padded_bin_img, cmap=plt.cm.bone)

        img_seg = np.multiply(ori_img, binary)
        plt.imsave(fname=save_dir + file_name + "_seg.png", arr=img_seg, cmap="gray")
        padded_seg_img = load_and_pad(save_dir + file_name + "_seg.png")
        plt.imsave(fname=save_dir + file_name + "_pad_seg.png", arr=padded_seg_img, cmap="gray")


if __name__ == "__main__":

    data_img_files = glob(RSNA_DATA_DIR + "*.png")
    if not os.path.isdir(RSNA_SEGMENT_SAVE_DIR):
        os.makedirs(RSNA_SEGMENT_SAVE_DIR)
    n_images = len(data_img_files)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_process', type=int, help="number of process to run code", default=4)
    FLAGS = parser.parse_args()
    num_process = FLAGS.num_process

    if num_process > 1:
        p = Pool()
        for i in range(num_process):
            if i < num_process-1:
                sub_image_files = data_img_files[i*(n_images//num_process): (i+1)*(n_images//num_process)]
            else:
                sub_image_files = data_img_files[i*(n_images//num_process):]
            p.apply_async(image_enhance_seg, args=(sub_image_files, (5, 5), False))

        p.close()
        p.join()
    else:
        image_enhance_seg(data_img_files)

