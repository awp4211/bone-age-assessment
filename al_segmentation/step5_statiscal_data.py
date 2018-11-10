import argparse
import pprint
import numpy as np
import os

from glob import glob
from utils_data import load_obj
from config import *


model_names = ["inception_v3", "inception_resnet_v2", "xception"]
sexs = ["0", "1", "2"]
fine_tunes = ["0", "3"]
augments = ["True"]
keys = ["evs", "mae", "mse", "meae", "r2s", "ccc"]


def best_value(data_dir, key="mae", min=True):
    metric_objs = glob(data_dir+"/*.pkl")
    values = []
    if len(metric_objs) != N_TRAINING_EPOCH:
        return 0.0
    for i in range(N_TRAINING_EPOCH):
        obj = load_obj(data_dir+"/epoch_{}.pkl".format(i))
        value = obj[key]
        values.append(value)
    if min:
        return np.min(values)
    else:
        return np.max(values)



for model_name in model_names:
    for fine_tune in fine_tunes:
        for sex in sexs:
            for augment in augments:
                metric_out_dir = "E{}_M{}_S{}_A{}_F{}/metric".format("regression2", model_name, sex, augment, fine_tune)
                # print metric_dir
                if os.path.isdir(metric_out_dir):
                    print "=========================================================================================="
                    print metric_out_dir
                    for key in keys:
                        print "{}, min: {}, max: {}".format(key,
                                                            best_value(metric_out_dir, key, True),
                                                            best_value(metric_out_dir, key, False))