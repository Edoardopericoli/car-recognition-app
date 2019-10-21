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

train = pd.read_csv("data/labels/train_labels.csv")
validation = pd.read_csv("data/labels/validation_labels.csv")
test = pd.read_csv("data/labels/test_labels.csv")


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