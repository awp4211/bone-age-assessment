import cv2
import argparse
import numpy as np
import os
import time

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_input

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_input

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet_50_input

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_input

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_input

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_input

from utils_data import load_obj
from step4_regression_with_cam import _build_regresser, load_sample_data

from tqdm import tqdm


########################################CAM#############################################################################
def predict_on_weights(out_base, weights):
    gap = np.average(out_base, axis=(0, 1))
    logit = np.dot(gap, np.squeeze(weights))
    return 1 / (1 +  np.e ** (-logit))


def getCAM(image, feature_maps, weights, plot_name=""):
    predict = predict_on_weights(feature_maps, weights)

    # Weighted Feature Map
    cam = (predict - 0.5) * np.matmul(feature_maps, weights)
    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    # Resize as image size
    cam_resize = cv2.resize(cam, (image.shape[0], image.shape[1]))
    # Format as CV_8UC1 (as applyColorMap required)
    cam_resize = 255 * cam_resize
    cam_resize = cam_resize.astype(np.uint8)
    # Get Heatmap
    heatmap = cv2.applyColorMap(cam_resize, cv2.COLORMAP_JET)
    # Zero out
    heatmap[np.where(cam_resize <= 100)] = 0

    image = (image+128)
    out = cv2.addWeighted(src1=image, alpha=0.8, src2=heatmap, beta=0.4, gamma=0)
    out = cv2.resize(out, dsize=(400, 400))

    if plot_name != "":
        cv2.imwrite(plot_name, out)
    return out


def batch_CAM(weights, data, base_model, original_int_images):
    idx = 0
    data_count = data.shape[0]
    result = None
    for j in range(int(np.sqrt(data_count))):
        for i in range(int(np.sqrt(data_count))):
            # src = data[idx][:, :, ::-1]
            src = data[idx]
            out_base = base_model.predict(np.expand_dims(src, axis=0))
            out_base = out_base[0]
            ori = original_int_images[idx]
            out = getCAM(image=ori, feature_maps=out_base, weights=weights)
            out = cv2.resize(out, dsize=(300, 300))
            if i > 0:
                canvas = np.concatenate((canvas, out), axis=1)
            else:
                canvas = out
            idx += 1
        if j > 0:
            result = np.concatenate((result, canvas), axis=0)
        else:
            result = canvas
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help="training batch size", default=32)
    parser.add_argument('--model_name', type=str, help="model name: inception_v3 ....", default="inception_v3")
    parser.add_argument('--exp', type=str, help="experiment name", default="cam")
    parser.add_argument('--sex', type=int, help="0 for all, 1 for male, 2 for female", default=1)
    parser.add_argument('--augment', type=str, help="augment data", default="false")
    parser.add_argument('--fine_tune', type=str, help="fine tune pretrained layer", default="3")
    parser.add_argument('--num_gpu', type=int, default=1)
    FLAGS = parser.parse_args()

    num_gpu = FLAGS.num_gpu
    batch_size = FLAGS.batch_size * num_gpu
    model_name = FLAGS.model_name
    exp_name = FLAGS.exp
    sex = FLAGS.sex
    augment_data = True if FLAGS.augment == "true" else False
    fine_tune = FLAGS.fine_tune

    metric_out_dir = "E{}_M{}_S{}_A{}_F{}/metric".format(exp_name, model_name, sex, augment_data, fine_tune)
    model_out_dir = "E{}_M{}_S{}_A{}_F{}/model".format(exp_name, model_name, sex, augment_data, fine_tune)
    cam_out_dir = "E{}_M{}_S{}_A{}_F{}/cam".format(exp_name, model_name, sex, augment_data, fine_tune)

    if not os.path.isdir(cam_out_dir):
        os.makedirs(cam_out_dir)

    if model_name == "inception_v3":
        preprocess_fn = inception_v3_input
    elif model_name == "inception_resnet_v2":
        preprocess_fn = inception_resnet_input
    elif model_name == "xception":
        preprocess_fn = xception_input
    else:
        raise ValueError("Not a supported model name")

    print "[x] load saved model file"
    weights_history = load_obj(model_out_dir + "/weights_history.pkl")
    model, input_shape, base_model = _build_regresser(model_name, weights="imagenet", num_gpu=1, fine_tune=fine_tune)
    base_model.load_weights(filepath=model_out_dir + "/base_model.h5")

    sampled_x, sampled_y, sampled_original_int_images = load_sample_data(sex=sex,
                                                                         img_size=input_shape[0],
                                                                         preprocess_fn=preprocess_fn)

    file_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())) + ".mp4"
    print "[x] saving file to {}/{}".format(cam_out_dir, file_name)
    file_path = "{}/{}".format(cam_out_dir, file_name)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(file_path, fourcc, 20.0, (1200, 1200))
    for weight in tqdm(weights_history):
        img = batch_CAM(weight, sampled_x, base_model, sampled_original_int_images)
        out.write(img)
    out.release()
    cv2.destroyAllWindows()
