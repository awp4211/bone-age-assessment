from config import *

templete = "python step4_regression_multi_gpu.py --exp=regression3 --batch_size={} --model_name={} --sex={} --augment={} --fine_tune={} --num_gpu={}"


num_gpu = 5
batch_size = 32
model_names = ["inception_v3", "inception_resnet_v2", "xception"]
sexs = ["0", "1", "2"]
# fine_tunes = ["0", "1", "2", "3"]
fine_tunes = ["0", "3"]
augments = ["true"]

count = 0
for model_name in model_names:
    for sex in sexs:
        for augment in augments:
            for fine_tune in fine_tunes:
                count += 1
                print templete.format(batch_size, model_name, sex, augment, fine_tune, num_gpu)

print count