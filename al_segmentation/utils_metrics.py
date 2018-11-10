import os
import sys

from skimage import filters
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib


def print_metrics(itr, **kargs):
    print "*** Round {}  ====> ".format(itr),
    for name, value in kargs.items():
        print ("{} : {}, ".format(name, value)),
    print ""
    sys.stdout.flush()


def threshold_by_otsu(preds, flatten=True):
    # cut by otsu threshold
    threshold = filters.threshold_otsu(preds)
    pred_bin = np.zeros(preds.shape)
    pred_bin[preds >= threshold] = 1

    if flatten:
        return pred_bin.flatten()
    else:
        return pred_bin

"""
def best_f1_threshold(precision, recall, thresholds):
    best_f1 = -1
    for index in range(len(precision)):
        curr_f1 = 2. * precision[index] * recall[index] / (precision[index] + recall[index])
        if best_f1 < curr_f1:
            best_f1 = curr_f1
            best_threshold = thresholds[index]

    return best_f1, best_threshold

def threshold_by_f1(true_vessels, generated, masks, flatten=True, f1_score=False):
    vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_vessels, generated, masks)
    precision, recall, thresholds = precision_recall_curve(vessels_in_mask.flatten(), generated_in_mask.flatten(),
                                                           pos_label=1)
    best_f1, best_threshold = best_f1_threshold(precision, recall, thresholds)

    pred_vessels_bin = np.zeros(generated.shape)
    pred_vessels_bin[generated >= best_threshold] = 1

    if flatten:
        if f1_score:
            return pred_vessels_bin[masks == 1].flatten(), best_f1
        else:
            return pred_vessels_bin[masks == 1].flatten()
    else:
        if f1_score:
            return pred_vessels_bin, best_f1
        else:
            return pred_vessels_bin


def dice_coefficient(true_vessels, pred_vessels, masks):
    thresholded_vessels = threshold_by_f1(true_vessels, pred_vessels, masks, flatten=False)

    true_vessels = true_vessels.astype(np.bool)
    thresholded_vessels = thresholded_vessels.astype(np.bool)

    intersection = np.count_nonzero(true_vessels & thresholded_vessels)

    size1 = np.count_nonzero(true_vessels)
    size2 = np.count_nonzero(thresholded_vessels)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc
"""


def dice_coefficient_in_train(true_vec, pred_vec):
    true_vec = true_vec.astype(np.bool)
    pred_vec = pred_vec.astype(np.bool)

    intersection = np.count_nonzero(true_vec & pred_vec)

    size1 = np.count_nonzero(true_vec)
    size2 = np.count_nonzero(pred_vec)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def misc_measures_in_train(gts_vec, preds_vec):
    gts_vec = gts_vec.astype(np.bool)
    preds_vec = preds_vec.astype(np.bool)

    cm = confusion_matrix(gts_vec, preds_vec)
    acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = 1. * cm[1, 1] / ((cm[1, 0] + cm[1, 1]) + 1.)
    specificity = 1. * cm[0, 0] / ((cm[0, 1] + cm[0, 0]) + 1.)
    return acc, sensitivity, specificity


def metric_all_value(gts, preds):
    """
    metric all value in training step
    :param gts: 4D array
    :param preds: 4D array
    :return:
    """
    assert len(preds.shape) == 4
    assert len(gts.shape) == 4

    gts = np.squeeze(gts, axis=-1)  # TO 3D array: [BATCH_SIZE, SIDE_LENGTH, SIDE_LENGTH]
    preds = np.squeeze(preds, axis=-1)  # TO 3D array: [BATCH_SIZE, SIDE_LENGTH, SIDE_LENGTH]

    gts_vec = gts.flatten()
    preds_vec = preds.flatten()

    binary_preds = threshold_by_otsu(preds, flatten=False)
    binary_preds_vec = binary_preds.flatten()

    dice_coeff = dice_coefficient_in_train(preds_vec, binary_preds_vec)
    acc, sensitivity, specificity = misc_measures_in_train(gts_vec, binary_preds_vec)

    return binary_preds, dice_coeff, acc, sensitivity, specificity