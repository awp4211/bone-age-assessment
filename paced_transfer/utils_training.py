import numpy as np
import cv2

from keras import optimizers
from keras.utils import multi_gpu_model


def _set_trainable_layers(model, start_layer):
    for i, layer in enumerate(model.layers):
        if i < start_layer:
            layer.trainable = False
        else:
            layer.trainable = True


def _pprint(content):
    if content.startswith("T") or content.startswith("N"):
        print content


def _batch_cvt(batch_data):
    imgs = []
    for data in batch_data:
        img = np.squeeze(data)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imgs.append(img)
    imgs = np.array(imgs, dtype=np.float32)
    return imgs
