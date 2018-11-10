import numpy as np
import matplotlib.pyplot as plt
from utils_data import load_obj
from config import *
from glob import glob


def get_result_CTL(sex, exp="CTL", start_layer=-1, img_size=299):
    dir = "E{}{}_S{}_IMG_{}".format(exp, start_layer, sex, img_size)
    print dir
    measure = load_obj(dir+"/metric/evaluate_S{}.pkl".format(start_layer))
    losses = load_obj(dir+"/metric/losses_S{}.pkl".format(start_layer))
    return losses, measure


def get_result_PTL(sex, exp="PTL", img_size=299):
    dir = "E{}_S{}_IMG_{}".format(exp,sex, img_size)
    start_layers = [-1, XCEPTION_EXIT_START, XCEPTION_MID_START, XCEPTION_ENTRY_START, 0]
    losses = []
    measures = []
    for start_layer in start_layers:
        loss_file = "{}/metric/losses_S{}.pkl".format(dir, start_layer)
        evaluate_file = "{}/metric/evaluate_S{}.pkl".format(dir, start_layer)
        losses.extend(load_obj(loss_file))
        measures.append(load_obj(evaluate_file))

    return losses, measures


def plot(sex):
    start_layers = [-1, XCEPTION_EXIT_START, XCEPTION_MID_START, XCEPTION_ENTRY_START, 0]
    labels = ["Fine-tuning From FC", "Fine-tuning From Exit Block", "Fine-tuning From Mid Block",
              "Fine-tuning From Entry Block", "Fine-tuning All Layers", "Paced Transfer Learning"]
    colors = ['r', 'b', 'y', 'g', '#7e7e7e', 'm']
    if sex == 0:
        sexT = "All"
    elif sex == 1:
        sexT = "Male"
    elif sex == 2:
        sexT = "Female"

    plt.title("Loss on {} cohort".format(sexT))
    plt.xlabel("training epochs")
    plt.ylabel("loss (MAE)")
    for i in range(5):
        losses, measures = get_result_CTL(sex=sex, exp="CTL", start_layer=start_layers[i])
        print "CTL, sex = {}, start_layer = {}, measures:{}".format(sex, start_layers[i], measures)
        plt.plot(np.arange(250), losses, label=labels[i], color=colors[i])

    losses, measures = get_result_PTL(sex=sex, exp="PTL")
    print "PTL, sex = {}, measures: {}".format(sex, measures)
    plt.plot(np.arange(250), losses, label=labels[5], color=colors[5])

    plt.grid(True)
    plt.grid(color='gray', linewidth='0.3', linestyle='--')
    plt.legend(loc="best")
    plt.savefig("plot/losses_{}.png".format(sex))
    plt.close()


if __name__ == "__main__":
    """
    start_layers = [-1, XCEPTION_EXIT_START, XCEPTION_MID_START, XCEPTION_ENTRY_START, 0]
    sexs = ["0", "1", "2"]
    for sex in sexs:
        for start_layer in start_layers:
            try:
                los, mea = get_result_CTL(sex=sex, exp="CTL", start_layer=start_layer)
                print los
            except:
                continue

    losses, measures = get_result_PTL(2, exp="PTL")
    print losses
    """
    sexs = [0, 1, 2]
    for sex in sexs:
        plot(sex)

