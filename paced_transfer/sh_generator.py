from config import *

templete = "python step1_paced_transfer_learning.py " \
           "--exp=regression " \
           "--img_size={} " \
           "--gpu_id={} " \
           "--sex={} " \
           "--batch_size={} " \
           "--start_layer={}"

templete2 = "python step1_ptl_multithread.py " \
           "--exp=RM " \
           "--img_size={} " \
           "--gpu_id={} " \
           "--sex={} " \
           "--batch_size={} " \
           "--start_layer={} " \
           "--data_thread_num={}"

templete_ptl = "python step1_ptl_gen.py " \
           "--exp={} " \
           "--img_size={} " \
           "--gpu_id={} " \
           "--sex={} " \
           "--batch_size={} " \
           "--start_layer={} " \
           "--n_epoch={} "


templete_ctl = "python step2_conventional_transfer.py " \
           "--exp={} " \
           "--img_size={} " \
           "--gpu_id={} " \
           "--sex={} " \
           "--batch_size={} " \
           "--start_layer={} " \
           "--n_epoch={} "

img_sizes = [299]
batch_size = 32
sexs = ["0", "1", "2"]
start_layers = [-1, XCEPTION_EXIT_START, XCEPTION_MID_START, XCEPTION_ENTRY_START, 0]

gpu_id = 7
n_epoch = 50
for sex in sexs:
    for img_size in img_sizes:
        for start_layer in start_layers:
            print templete_ptl.format("PTL", img_size, gpu_id, sex, batch_size, start_layer, n_epoch)

gpu_id = 6
n_epoch = 250
for sex in sexs:
    for img_size in img_sizes:
        for start_layer in start_layers:
            print templete_ctl.format("CTL{}".format(start_layer), img_size, gpu_id, sex, batch_size, start_layer, n_epoch)