import numpy as np

from glob import glob

import utils_data
from config import *


def get_miss_ids():
    data_dir = "./interactive_segmentation_dice1.0_pixel1.0_GPU0/segmented_all"

    data_files = glob(data_dir + "/*_crop_seg.png")

    existed_ids = []
    for data_file in data_files:
        data_id = data_file[data_file.rfind("/") + 1: data_file.find("_crop")]
        existed_ids.append(data_id)

    all_ids = []
    data_files = glob(RSNA_DATA_DIR + "/*.png")
    for data_file in data_files:
        data_id = data_file[data_file.rfind("/") + 1: data_file.find(".png")]
        all_ids.append(data_id)

    for id in existed_ids:
        all_ids.remove(id)

    return all_ids


def check_all():
    missed_ids = get_miss_ids()
    if len(missed_ids) == 0:
        print "!!!!"
    else:
        print missed_ids

if __name__ == "__main__":
    check_all()