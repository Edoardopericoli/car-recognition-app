import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from math import ceil
import click
import logging
import os
import yaml
from utils.main_utils import asserting_batch_size


@click.command()
@click.option('--initial_parameters_path', default=r"config\initial_parameters.yml", help='config file containing initial parameters')
def main(initial_parameters_path):
    logging.info('Starting the process')
    logging.info('Asserting dimensions of train, validation and test')

    # Asserting that dimensions of train, validation and test are consistent
    full_data_length = len(os.listdir('../data/raw_data/cars_train'))
    train_length = len(os.listdir('../data/train'))
    validation_length = len(os.listdir('../data/validation'))
    test_length = len(os.listdir('../data/test'))
    assert full_data_length == train_length + validation_length + test_length

    logging.info('Loading data')
    train_df = pd.read_csv("data/labels/train_labels.csv")
    validation_df = pd.read_csv("data/labels/validation_labels.csv")
    test_df = pd.read_csv("data/labels/test_labels.csv")

    with open(initial_parameters_path) as file:
        initial_parameters = yaml.load(file)

    logging.info('Asserting batch sizes')
    # Asserting dimensions of batch sizes
    asserting_batch_size(length_data=len(train_df), batch_size=initial_parameters['train_batch_size'])
    asserting_batch_size(length_data=len(validation_df), batch_size=initial_parameters['validation_batch_size'])
    asserting_batch_size(length_data=len(test_df), batch_size=initial_parameters['test_batch_size'])

    logging.info('Transforming data using ImageDataGenerator')
    train_image_generator = ImageDataGenerator(rescale=1./255)
    validation_image_generator = ImageDataGenerator(rescale=1./255)
    test_image_generator = ImageDataGenerator(rescale=1./255)

    train_generator = train_image_generator.flow_from_dataframe(
        dataframe=train_df,
        directory="data/train",
        x_col="fname",
        y_col=initial_parameters['classes'],
        batch_size=initial_parameters['train_batch_size'],
        seed=initial_parameters['seed'],
        shuffle=True,
        class_mode="other",
        target_size=(initial_parameters['IMG_HEIGHT'], initial_parameters['IMG_WIDTH'])
    )

    validation_generator = validation_image_generator.flow_from_dataframe(
        dataframe=validation_df,
        directory="data/validation",
        x_col="fname",
        y_col=initial_parameters['classes'],
        batch_size=initial_parameters['validation_batch_size'],
        seed=initial_parameters['seed'],
        shuffle=True,
        class_mode="other",
        target_size=(initial_parameters['IMG_HEIGHT'], initial_parameters['IMG_WIDTH'])
    )

    test_generator = test_image_generator.flow_from_dataframe(
        dataframe=test_df,
        directory="data/test",
        x_col="fname",
        y_col=initial_parameters['classes'],
        batch_size=initial_parameters['test_batch_size'],
        seed=initial_parameters['seed'],
        shuffle=True,
        class_mode="other",
        target_size=(initial_parameters['IMG_HEIGHT'], initial_parameters['IMG_WIDTH'])
    )

    #Model
    model = Sequential([
        Conv2D(1, 8, activation='relu', input_shape=(initial_parameters['IMG_HEIGHT'], initial_parameters['IMG_WIDTH'], 3)),
        MaxPooling2D((8, 8)),
        Conv2D(1, 8, padding='same', activation='relu'),
        MaxPooling2D((8, 8)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

# TODO: Togliere test dai dataset caricati
# TODO: Agiungere file per fare evaluation finale
# TODO: Creare setting per scegliere se fare gridsearch

    logging.info('Fitting the model')
    history = model.fit_generator(
        train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen,
        steps_per_epoch=ceil(83484 / tr_batch_size)
     )

    # Saving model
    model_names = [name for name in os.listdir(cwd + '/models')]
    if len(model_names) == 0:
        model_name = '1'
        os.mkdir(cwd + '/models/' + model_name)
        yaml_string = model.to_yaml()
        with open(cwd + '/models/' + model_name + '/architecture.yml', 'w') as outfile:
            yaml.dump(yaml_string, outfile, default_flow_style=False)
        # Saving global parameters
        with open(cwd + '/models/' + model_name + '/parameters.yml', 'w') as outfile:
            yaml.dump(global_parameters, outfile, default_flow_style=False)
        # Saving estimator
        #model.save(cwd + '/models/' + model_name + '/model.h5')
    else:
        model_name = str(int(model_names[-1])+1)
        os.mkdir(cwd + '/models/'+ model_name)
        # Saving architecture
        yaml_string = model.to_yaml()
        with open(cwd + '/models/' + model_name + '/architecture.yml', 'w') as outfile:
            yaml.dump(yaml_string, outfile, default_flow_style=False)
        # Saving global parameters
        with open(cwd + '/models/' + model_name + '/parameters.yml', 'w') as outfile:
            yaml.dump(global_parameters, outfile, default_flow_style=False)
        # Saving estimator
        model.save(cwd + '/models/' + model_name + '/model.h5')

        # Model Performance
        # acc = history.history['acc']
        # val_acc = history.history['val_accuracy']

        # loss = history.history['loss']
        # val_loss = history.history['val_loss']

if __name__ == '__main__':
    main()