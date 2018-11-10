import numpy as np
import os
import argparse
import cv2
import utils_data

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

from config import *


def image_to_numpy(image_file, image_id, type="x", save_dir=RSNA_GT_NP_ANNOTATED):
    assert type in ["x", "y"]
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if img.shape[0] > img.shape[1]:
        pad = (img.shape[0] - img.shape[1]) // 2
        pad_tuple = ((0, 0), (pad, pad))
    else:
        pad = (img.shape[1] - img.shape[0]) // 2
        pad_tuple = ((pad, pad), (0, 0))
    img = np.pad(img, pad_tuple, mode="constant")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    if type == "x":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    if type == "y":
        img[img > 0] = 255
    img = img / 255.
    img = np.array(img, dtype=np.float32)
    utils_data.write_hdf5(img, save_dir + "/{}_{}.h5".format(image_id, type))


def save_images(image_files, type="x"):
    for f in tqdm(image_files):
        image_id = f[f.rfind("/") + 1: f.find("_")]
        image_to_numpy(f, image_id, type)


if __name__ == "__main__":
    if not os.path.isdir(RSNA_GT_NP_ANNOTATED):
        os.makedirs(RSNA_GT_NP_ANNOTATED)
    if not os.path.isdir(RSNA_GT_NP_UNANNOTATED):
        os.makedirs(RSNA_GT_NP_UNANNOTATED)
    if not os.path.isdir(RSNA_GT_NEW):
        os.makedirs(RSNA_GT_NEW)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_process', type=int, help="number of process to run code", default=4)
    parser.add_argument('--annotate', type=str, help="weather to save annotated or unannotated data", default="adddata")
    FLAGS = parser.parse_args()

    num_process = FLAGS.num_process
    annotate = FLAGS.annotate

    if annotate == "annotate":
        # initial annotated image data
        data_x_files = glob(RSNA_GT + "/*_ori.png")
        data_y_files = glob(RSNA_GT + "/*_bin.png")
        n_images = len(data_x_files)
        print "[x] start to pre processing %d images" % n_images

        if num_process > 1:
            p = Pool()
            for i in range(num_process):
                if i < num_process-1:
                    sub_image_files_x = data_x_files[i*(n_images//num_process): (i+1)*(n_images//num_process)]
                    sub_image_files_y = data_y_files[i*(n_images//num_process): (i+1)*(n_images//num_process)]
                else:
                    sub_image_files_x = data_x_files[i*(n_images//num_process):]
                    sub_image_files_y = data_y_files[i*(n_images//num_process):]
                p.apply_async(save_images, args=(sub_image_files_x, "x"))
                p.apply_async(save_images, args=(sub_image_files_y, "y"))
            p.close()
            p.join()
        else:
            save_images(data_x_files, "x")
            save_images(data_y_files, "y")
    elif annotate == "unannotate":
        # initial unannotated data
        import utils_data

        annotated_ids = utils_data.load_annotated_data_np(data_dir=RSNA_GT_NP_ANNOTATED, return_ids=True)
        data_x_files = glob(RSNA_DATA_DIR + "/*.png")
        for f in tqdm(data_x_files):
            data_id = f[f.rfind("/")+1: f.find(".png")]
            if data_id not in annotated_ids:
                image_to_numpy(f, data_id, type="x", save_dir=RSNA_GT_NP_UNANNOTATED)

    elif annotate == "adddata":
        data_y_files = glob(RSNA_GT_NEW + "/*_bin.png")
        n_images = len(data_y_files)
        print "[x] start to add %d images" % n_images
        for f in data_y_files:
            data_id = f[f.rfind("/")+1: f.find("_bin")]
            print "[x] add data %s " % data_id
            image_x_file = RSNA_GT_NEW + "/{}.png".format(data_id)
            image_to_numpy(image_x_file, data_id, type="x", save_dir=RSNA_GT_NP_ANNOTATED)
            image_to_numpy(f, data_id, type="y", save_dir=RSNA_GT_NP_ANNOTATED)


"""
Script to initialize
python step2_image_2_h5.py --annotate=annotate --num_process=4
python step2_image_2_h5.py --annotate=unannotate

Script to add data
python step2_image_2_h5.py --annotate=adddata
"""