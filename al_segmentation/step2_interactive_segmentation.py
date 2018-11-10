import numpy as np
import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import heapq
import time

import utils_data

from tqdm import tqdm
from skimage.color import rgb2grey
from skimage import filters

from tf_model_discriminator import pixel_discriminator
from tf_model_dense import dense_net
from tf_model_unet import unet
from tf_ops import dice_coef_loss, pixelwise_cross_entropy

from utils_metrics import metric_all_value, threshold_by_otsu
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help="training batch size", default=4)
parser.add_argument('--init_lr', type=float, help="initial learning rate", default=1E-4)
parser.add_argument('--dice', type=float, help="dice coefficient loss ratio", default=1.0)
parser.add_argument('--pixel', type=float, help="pixelwise cross entropy loss ratio", default=1.0)
parser.add_argument('--gpu', type=int, help="GPU number", default=0)
parser.add_argument('--exp', type=str, help="experiment name", default="interactive_segmentation")
FLAGS = parser.parse_args()

batch_size = FLAGS.batch_size
dice_ratio = FLAGS.dice
pixel_ratio = FLAGS.pixel
init_lr = FLAGS.init_lr
gpu = FLAGS.gpu

experiment_name = "{}_dice{}_pixel{}_GPU{}".format(FLAGS.exp, dice_ratio, pixel_ratio, gpu)
img_out_dir = "{}/segmentation_results".format(experiment_name)
model_out_dir = "{}/model".format(experiment_name)
metrics_out_dir = "{}/metrics".format(experiment_name)

if not os.path.isdir(img_out_dir):
    os.makedirs(img_out_dir)
if not os.path.isdir(model_out_dir):
    os.makedirs(model_out_dir)
if not os.path.isdir(metrics_out_dir):
    os.makedirs(metrics_out_dir)


print "[x] building models on GPU: {}".format(gpu)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# create model 0
is_train = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32)
print "[x] building Model 0"
tf.set_random_seed(1377)
x_0 = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], name="x_0")
y_0 = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], name="y_0")
y_prob0, conv9_feature0 = unet(x_0, is_train, n_filters=32, name="u-net_0")
dice_coef_loss_0 = dice_coef_loss(y_prob0, y_0)
pixelwise_cross_entropy_0 = pixelwise_cross_entropy(y_prob0, y_0)
loss_0 = dice_ratio * dice_coef_loss_0 + pixel_ratio * pixelwise_cross_entropy_0
train_op_0 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_0)

print "[x] building Model 1"
tf.set_random_seed(7988)
x_1 = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], name="x_1")
y_1 = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], name="y_1")
y_prob1, conv9_feature1 = unet(x_1, is_train, n_filters=32, name="u-net_1")
dice_coef_loss_1 = dice_coef_loss(y_prob1, y_1)
pixelwise_cross_entropy_1 = pixelwise_cross_entropy(y_prob1, y_1)
loss_1 = dice_ratio * dice_coef_loss_1 + pixel_ratio * pixelwise_cross_entropy_1
train_op_1 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_1)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

print "[x] start training"
step = 0
current_lr = init_lr
last_operator = "a"
for epoch in tqdm(range(N_TRAINING_EPOCH)):
    print "=================================================="
    print "[x] epoch {}, training U-Net ...".format(epoch)
    print "[x] epoch {}, loading training data".format(epoch)
    annotated_ids = utils_data.load_annotated_data_np(data_dir=RSNA_GT_NP_ANNOTATED, return_ids=True)
    print "[x] load annotated data ..."
    annotated_x, annotated_y, annotated_ids = utils_data.load_annotated_data_np(data_dir=RSNA_GT_NP_ANNOTATED, return_ids=False)
    print "[x] load unannotated data ..."
    unannotated_x, unannotated_ids = utils_data.load_unannotated_data_np(data_dir=RSNA_GT_NP_UNANNOTATED, annotated_ids=annotated_ids)
    print "[x] loaded annotated data: {}, unannotated data: {}".format(annotated_x.shape[0], unannotated_x.shape[0])
    if epoch == N_TRAINING_EPOCH/2:
        current_lr = 0.5 * init_lr
    if epoch == int(0.75*N_TRAINING_EPOCH):
        current_lr = 0.5 * init_lr

    shuffled_x, shuffled_y, shuffled_indexes = utils_data.shuffle_data(data_x=annotated_x, data_y=annotated_y)
    for mini_batch in range(annotated_x.shape[0]//batch_size):
        batch_x = shuffled_x[mini_batch*batch_size: (mini_batch+1)*batch_size]
        batch_y = shuffled_y[mini_batch*batch_size: (mini_batch+1)*batch_size]
        [batch_loss_0, _] = sess.run([loss_0, train_op_0],
                                     feed_dict={x_0: batch_x, y_0: batch_y, is_train: True, learning_rate: current_lr})
        [batch_loss_1, _] = sess.run([loss_1, train_op_1],
                                     feed_dict={x_1: batch_x, y_1: batch_y, is_train: True, learning_rate: current_lr})
        if step % 100 == 0:
            print "[--] epoch: {}, mini_batch: {}, global step: {}, loss_0: {}, loss_1: {}". \
                format(epoch, mini_batch, step, batch_loss_0, batch_loss_1)
            [batch_segmented_0] = sess.run([y_prob0], feed_dict={x_0: batch_x, is_train: False})
            [batch_segmented_1] = sess.run([y_prob1], feed_dict={x_1: batch_x, is_train: False})
            binary_preds_0, dice_coeff_0, acc_0, sensitivity_0, specificity_0 = metric_all_value(batch_y, batch_segmented_0)
            binary_preds_1, dice_coeff_1, acc_1, sensitivity_1, specificity_1 = metric_all_value(batch_y, batch_segmented_1)
            utils_data.save_obj({"step": step, "batch_loss_0": batch_loss_0, "batch_loss_1": batch_loss_1,
                "dice_coeff_0": dice_coeff_0, "acc_0": acc_0, "sensitivity_0": sensitivity_0, "specificity_0": specificity_0,
                "dice_coeff_1": dice_coeff_1, "acc_1": acc_1, "sensitivity_1": sensitivity_1, "specificity_1": specificity_1},
                name=metrics_out_dir+"/step_{}_loss.pkl".format(step))

            if step % 500 == 0:
                output_dir = img_out_dir + "/step_{}".format(step)
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                for i in range(batch_size):
                    data_index = shuffled_indexes[mini_batch*batch_size+i]
                    data_id = annotated_ids[data_index]
                    plt.imsave(fname=output_dir+"/{}_padded_ori.png".format(data_id), arr=np.squeeze(batch_x[i], axis=2), cmap="gray")
                    plt.imsave(fname=output_dir+"/{}_padded_model_0_bin.png".format(data_id), arr=binary_preds_0[i], cmap="gray")
                    plt.imsave(fname=output_dir+"/{}_padded_model_1_bin.png".format(data_id), arr=binary_preds_1[i], cmap="gray")
                    plt.imsave(fname=output_dir+"/{}_padded_bin.png".format(data_id), arr=np.squeeze(batch_y[i], axis=2), cmap="gray")
        step += 1

    print "-------------------Test On Unannotated Data---------------------"
    output_dir = img_out_dir + "/epoch_{}".format(epoch)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    n_unannotated_data = unannotated_x.shape[0]
    selected_indexes = np.random.permutation(n_unannotated_data)[:batch_size*TEST_NUM]
    selected_data_x = unannotated_x[selected_indexes]
    for mini_batch in range(TEST_NUM):
        batch_x = selected_data_x[mini_batch*batch_size: (mini_batch+1)*batch_size]
        [model_0_prob] = sess.run([y_prob0], feed_dict={x_0: batch_x, is_train: False})
        [model_1_prob] = sess.run([y_prob1], feed_dict={x_1: batch_x, is_train: False})
        for i in range(batch_size):
            image_index = mini_batch*batch_size+i
            plt.imsave(fname=output_dir+"/{}_ori.png".format(image_index), arr=np.squeeze(batch_x[i]), cmap="gray")
            plt.imsave(fname=output_dir+"/{}_model_0.png".format(image_index), arr=np.squeeze(model_0_prob[i]), cmap="gray")
            plt.imsave(fname=output_dir+"/{}_model_1.png".format(image_index), arr=np.squeeze(model_1_prob[i]), cmap="gray")
            plt.imsave(fname=output_dir+"/{}_model_0_bin.png".format(image_index),
                       arr=threshold_by_otsu(np.squeeze(model_0_prob[i]), flatten=False), cmap="gray")
            plt.imsave(fname=output_dir+"/{}_model_1_bin.png".format(image_index),
                       arr=threshold_by_otsu(np.squeeze(model_1_prob[i]), flatten=False), cmap="gray")

    print "-------------Interactive Segmentation --------------------------"
    unannotated_imgs_similarity = []
    # 1. Calculate each images similarity between the two U-Net Models
    print "[x] inference all unannotated images and calculate uncertainty ..."
    for mini_batch in range(unannotated_x.shape[0]//batch_size):
        batch_x = unannotated_x[mini_batch*batch_size: (mini_batch+1)*batch_size]
        [feature_0] = sess.run([conv9_feature0], feed_dict={x_0: batch_x, is_train: False})
        [feature_1] = sess.run([conv9_feature1], feed_dict={x_1: batch_x, is_train: False})
        for i in range(batch_size):
            cosine_similarity = utils_data.similarity_cosine(feature_0[i], feature_1[i])
            unannotated_imgs_similarity.append(cosine_similarity)

    # 2. Select AL_UNCERTAINTY_NUM patches with lowest cosine similarity (uncertainty to annotate)
    lowest_similaritites = heapq.nsmallest(AL_UNCERTAINTY_NUM, unannotated_imgs_similarity)
    ids_of_lowest_similarity = []
    for l in lowest_similaritites:
        if l < AL_UNCERTAINTY_NUM:
            ids_of_lowest_similarity.append(unannotated_ids[unannotated_imgs_similarity.index(l)])

    # 3. Ask oracle annotation
    print "#Epoch {}######################".format(epoch)
    print "Image Ids to be annotated:"
    print ids_of_lowest_similarity

    print "[x] interactive waiting ..."
    if last_operator != "f":
        operator = raw_input(">")
        if operator == "s":
            # save_model
            saver.save(sess, model_out_dir+"/model.ckpt", global_step=step)
        elif operator == "f":
            last_operator = "f"
    else:
        saver.save(sess, model_out_dir + "/model.ckpt", global_step=step)

