import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from math import ceil
import yaml
import os

df = pd.read_csv("data/labels/labels.csv")
percentage_train = 0.8
percentage_validation = 0.1
percentage_test = 0.05

# TODO: changing the setting fro train test split --> using a predetermined function, POSSIBLY STRATIFYING THEM
##############################################################################
seed = 100
train_index = np.random.randint(0, len(df), round(percentage_train*len(df), 0), seed=seed)
validation_index = np.random.randint(0, len(df), round(percentage_validation*len(df), 0), seed=seed)
test_index = np.random.randint(0, len(df), round(percentage_test*len(df), 0), seed=seed)

df_train = df[train_index]
df_validation = df[validation_index]
df_test = df[test_index]
##############################################################################

classes = np.arange(1, 197)

train_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_image_generator.flow_from_dataframe(
    dataframe=df[:1800],
    directory="./miml_dataset/images",
    x_col="Filenames",
    y_col=columns,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(100,100)
)

valid_generator = test_image_generator.flow_from_dataframe(
    dataframe=df[1800:1900],
    directory="./miml_dataset/images",
    x_col="Filenames",
    y_col=columns,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(100,100)
)

test_generator = test_image_generator .flow_from_dataframe(
    dataframe=df[1900:],
    directory="./miml_dataset/images",
    x_col="Filenames",
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(100,100)
)






























def main():
    pass
if __name__ == '__main__':
    main()