# RSNA DATA
RSNA_DATA_DIR = "../data/boneage-training-dataset/"
RSNA_TRAIN_CSV = "../data/csv/train.csv"
RSNA_GT = "../data/gt/annotated"
RSNA_GT_NP_ANNOTATED = "../data/gt/annotated/np"
RSNA_GT_NP_UNANNOTATED = "../data/gt/unannotated"
RSNA_GT_NEW = "../data/gt/new"

# pre processing
RSNA_SEGMENT_SAVE_DIR = "../data/segmented_hand_al/"
RSNA_SEG_ENHANCE = "../data/segmented_enhance"
THRESH_RATIO = 0.9
IMAGE_SIZE = 256


# transfer learning RSNA
INCEPTION_V4_MODEL = "../data/pretrained_model/inception-v4_weights_tf_dim_ordering_tf_kernels.h5"
RSNA_TF_VGG16_PATH = "../transfer_features/vgg16/"
RSNA_TF_VGG19_PATH = "../transfer_features/vgg19/"
RSNA_TF_INCEPTION_V3_PATH = "../transfer_features/inceptionv3/"
RSNA_TF_INCEPTION_V4_PATH = "../transfer_features/inceptionv4/"
RSNA_TF_RESNET50_PATH = "../transfer_features/resnet50/"
RSNA_TF_XCEPTION_PATH = "../transfer_features/xception/"

# model
DEBUG_MODEL = False
N_TRAINING_EPOCH = 100
AL_UNCERTAINTY_NUM = 10
TEST_NUM = 20
SCALE = 240.
TOLERANCE = 5

FINE_TUNE_ALL = "3"
# Finetune - INCEPTION V3
INCEPTION_V3_INCEPTION_3 = "0"
INCEPTION_V3_INCEPTION_3_START = 40
INCEPTION_V3_INCEPTION_4 = "1"
INCEPTION_V3_INCEPTION_4_START = 87
INCEPTION_V3_INCEPTION_5 = "2"
INCEPTION_V3_INCEPTION_5_START = 229

# Finetune - INCEPTION-RES V2
INCEPTION_RESNET_V2_INCEPTION_A = "0"
INCEPTION_RESNET_V2_INCEPTION_A_START = 40
INCEPTION_RESNET_V2_INCEPTION_B = "1"
INCEPTION_RESNET_V2_INCEPTION_B_START = 275
INCEPTION_RESNET_V2_INCEPTION_C = "2"
INCEPTION_RESNET_V2_INCEPTION_C_START = 618

# Finetune Xception
XCEPTION_ENTRY = "0"
XCEPTION_ENTRY_START = 15
XCEPTION_MID = "1"
XCEPTION_MID_START = 35
XCEPTION_EXIT = "2"
XCEPTION_EXIT_START = 116
