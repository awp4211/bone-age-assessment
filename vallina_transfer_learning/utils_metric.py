import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, explained_variance_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

from sklearn.metrics import accuracy_score, confusion_matrix

def regression_metric(y_true, y_pred):
    # explained_variance_score
    evs = explained_variance_score(y_true, y_pred)
    # mean_absolute_error
    mae = mean_absolute_error(y_true, y_pred)
    # mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    # median_absolute_error
    meae = median_absolute_error(y_true, y_pred)
    # r^2_score
    r2s = r2_score(y_true, y_pred)
    ccc = _ccc(y_true, y_pred)
    return evs, mae, mse, meae, r2s, ccc


def _ccc(y_true, y_pred):
    x_mean = np.average(y_true)
    y_mean = np.average(y_pred)
    n = y_true.shape[0]
    s_xy = np.sum(np.multiply(y_true-x_mean, y_pred-y_mean)) / n
    s_x2 = np.sum([np.power(e, 2) for e in (y_true - x_mean)]) / n
    s_y2 = np.sum([np.power(e, 2) for e in (y_pred - y_mean)]) / n
    return 2*s_xy / (s_x2+s_y2+np.power(x_mean-y_mean, 2))


def AUC_ROC(true_vessel_arr, pred_vessel_arr):
    """
    Area under the ROC curve with x axis flipped
    """
    fpr, tpr, _ = roc_curve(true_vessel_arr, pred_vessel_arr)
    AUC_ROC = roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    return AUC_ROC, fpr, tpr


def AUC_PR(true_vessel_img, pred_vessel_img):
    """
    Precision-recall curve
    """
    try:
        precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(),  pos_label=1)
        AUC_prec_rec = auc(recall, precision)
        return AUC_prec_rec, precision, recall
    except:
        return 0.


def classify_metrics(y_true, y_pred):
    """
        cm = confusion_matrix(y_true, y_pred)
        acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
        sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
        specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
        return acc, sensitivity, specificity
        """
    acc = accuracy_score(y_true, y_pred)
    return acc

