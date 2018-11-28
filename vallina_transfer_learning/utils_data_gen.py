import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from config import *


def _build_img_generator(model_name="inception_v3"):
    from keras.preprocessing.image import ImageDataGenerator
    if model_name == "inception_v3":
        from keras.applications.inception_v3 import preprocess_input
    elif model_name == "inception_resnet_v2":
        from keras.applications.inception_resnet_v2 import preprocess_input
    elif model_name == "xception":
        from keras.applications.xception import preprocess_input
    else:
        raise ValueError("Not support {}".format(model_name))

    core_idg = ImageDataGenerator(samplewise_center=False,
                                  samplewise_std_normalization=False,
                                  horizontal_flip=True,
                                  vertical_flip=False,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  rotation_range=5,
                                  shear_range=0.01,
                                  fill_mode='nearest',
                                  zoom_range=0.25,
                                  preprocessing_function=preprocess_input)

    return core_idg


def _flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode='sparse', **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


def _load_data(sex=0, img_path=RSNA_TRAIN_DATA, num_sample_per_category=500):
    if sex == 0:
        dataset_df = pd.read_csv(RSNA_TRAIN_CSV)
    elif sex == 1:
        dataset_df = pd.read_csv(RSNA_TRAIN_CSV)
        dataset_df = dataset_df[dataset_df.male==True]
    else:
        dataset_df = pd.read_csv(RSNA_TRAIN_CSV)
        dataset_df = dataset_df[dataset_df.male==False]

    dataset_df['path'] = dataset_df['id'].map(lambda x: os.path.join(img_path ,'{}.png'.format(x)))
    dataset_df['exists'] = dataset_df['path'].map(os.path.exists)
    print(dataset_df['exists'].sum(), 'images found of', dataset_df.shape[0], 'total')

    boneage_mean = dataset_df['boneage'].mean()
    boneage_std = 2*dataset_df['boneage'].std()

    print "dataset mean: {}, dataset std: {}".format(boneage_mean, boneage_std)
    dataset_df['boneage_zscore'] = dataset_df['boneage'].map(lambda x: (x-boneage_mean)/boneage_std)
    dataset_df.dropna(inplace=True)
    dataset_df['boneage_category'] = pd.cut(dataset_df['boneage'], 20)

    train_df, validation_df = train_test_split(dataset_df, test_size=0.25, random_state=2018, stratify=dataset_df['boneage_category'])
    new_train_df = train_df.groupby(['boneage_category']).apply(lambda x: x.sample(num_sample_per_category, replace=True)).reset_index(drop=True)
    print "new data size : {}, old data size: {}".format(new_train_df.shape[0], train_df.shape[0])

    return train_df, validation_df, boneage_mean, boneage_std


def build_data_generator(model_name="inception_v3",
                         sex=0,
                         img_path=RSNA_TRAIN_DATA,
                         num_per_category=500,
                         img_size=512,
                         batch_size=32):
    img_generator = _build_img_generator(model_name)
    img_sizes = (img_size, img_size)
    train_data_df, validation_data_df, mean, std = _load_data(sex=sex,
                                                              img_path=img_path,
                                                              num_sample_per_category=num_per_category)
    train_gen = _flow_from_dataframe(img_generator, train_data_df,
                                     path_col='path',
                                     y_col='boneage_zscore',
                                     target_size=img_sizes,
                                     color_mode='rgb',
                                     batch_size=batch_size)

    valid_gen = _flow_from_dataframe(img_generator, validation_data_df,
                                     path_col='path',
                                     y_col='boneage_zscore',
                                     target_size=img_sizes,
                                     color_mode='rgb',
                                     batch_size=batch_size)  # we can use much larger batches for evaluation
    test_X, test_Y = next(_flow_from_dataframe(img_generator,
                                               validation_data_df,
                                               path_col='path',
                                               y_col='boneage_zscore',
                                               target_size=img_sizes,
                                               color_mode='rgb',
                                               batch_size=1000))  # one big batch
    return train_gen, valid_gen, mean, std, test_X, test_Y
