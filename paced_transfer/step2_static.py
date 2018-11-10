import numpy as np
import argparse

from glob import glob
from utils_data import load_hdf5, load_obj

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help="experiment name", default="regression")
parser.add_argument('--img_size', type=int, help="image size", default=299)
parser.add_argument('--sex', type=int, help="0 for all, 1 for male, 2 for female", default=0)
FLAGS = parser.parse_args()

exp_name = FLAGS.exp
img_size = FLAGS.img_size
sex = FLAGS.sex

metric_out_dir = "E{}_S{}_IMG_{}/metric".format(exp_name, sex, img_size)

metric_files = glob(metric_out_dir+"/*.*")
for i in range(len(metric_files)):
    metric_file = metric_out_dir + "/epoch_{}.pkl".format(i+1)
    obj =  load_obj(metric_file)
    print obj
