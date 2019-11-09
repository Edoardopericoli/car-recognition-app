import logging
import numpy as np
import pandas as pd
import yaml
import os
import csv
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras import optimizers
from keras.layers import GlobalAveragePooling2D, BatchNormalization
from keras import Model
import efficientnet.keras as efn
from math import ceil


def setting_log():
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def bound_cpu(n_threads=8):
    K.set_session(K.tf.Session(
                   config=K.tf.ConfigProto
                   (intra_op_parallelism_threads=n_threads,
                    inter_op_parallelism_threads=n_threads)))


def load_parameters(parameters_path):
    with open(parameters_path) as f:
        initial_parameters = yaml.load(f)
    return initial_parameters


def load_labels_dfs():
    file_path = os.path.dirname(os.path.abspath(__file__))
    train_df = pd.read_csv(file_path + "/../data/labels/train_labels.csv")
    validation_df = pd.read_csv(file_path + "/../data/labels/validation_labels.csv")
    train_df.reset_index(inplace=True)
    validation_df.reset_index(inplace=True)
    target_variable = 'model_label'
    train_df[target_variable] = train_df[target_variable].astype('str')
    validation_df[target_variable] = validation_df[target_variable].astype('str')
    return train_df, validation_df


def get_image_generators():
    train_image_generator = ImageDataGenerator(
                            rescale=1. / 255,
                            zoom_range=0.2,
                            rotation_range=5,
                            horizontal_flip=True)
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)
    return train_image_generator, validation_image_generator


def get_generator(imagedatagenerator, labels_df, directory,
                  initial_parameters, train=True):
    if train:
        batch_size = initial_parameters['train_batch_size']
    else:
        batch_size = initial_parameters['validation_batch_size']

    file_path = os.path.dirname(os.path.abspath(__file__))
    directory = file_path + '/' + directory
    train_generator = imagedatagenerator.flow_from_dataframe(
        dataframe=labels_df,
        directory=directory,
        x_col='fname',
        y_col='model_label',
        batch_size=batch_size,
        seed=initial_parameters['seed'],
        class_mode="categorical",
        target_size=(initial_parameters['IMG_HEIGHT'],
                     initial_parameters['IMG_WIDTH']),
    )
    return train_generator


def setup_base_model():
    base_model = efn.EfficientNetB1(weights='imagenet', include_top=False)
    # fix the feature extraction part of the model
    for layer in base_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
    return base_model


def setup_final_layers(base_model, train_generator):
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(len(train_generator.class_indices),
                        activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(lr=0.01),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def train_model(train_generator, validation_generator, initial_parameters,
                train_df, model):
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=ceil(len(train_df) /
                                                       initial_parameters
                                                       ['train_batch_size']),
                                  validation_steps=ceil(len(train_df) /
                                                        initial_parameters
                                                        ['validation_batch_size']
                                                        ),
                                  validation_data=validation_generator,
                                  epochs=10,
                                  workers=8,
                                  max_queue_size=32,
                                  verbose=1)
    return history


def save_model_architecture(username, model, initial_parameters):
    file_path = os.path.dirname(os.path.abspath(__file__))
    path = file_path + '/../data/models'
    model_names = [name for name in os.listdir(path)
                   if name.startswith(username)]
    if len(model_names) == 0:
        model_name = username + '_' + '1'
        logging.info('Saving: model, initial parameters \
                      and architecture into {model_directory}'.format(
                      model_directory=path + '/' + model_name))
        os.mkdir(path + '/' + model_name)
        yaml_string = model.to_yaml()
        with open(path + '/' + model_name + '/architecture.yml',
                  'w') as outfile:
            outfile.write(yaml_string)
        # Saving global parameters
        with open(path + '/' + model_name + '/initial_parameters.yml',
                  'w') as outfile:
            yaml.dump(initial_parameters, outfile, default_flow_style=False)
        # Saving estimator
        model.save(path + '/' + model_name + '/model.h5')
    else:
        model_name = username + '_' + str(int(model_names[-1].split('_')[-1])
                                          + 1)
        logging.info('Saving: model, initial parameters and \
                      architecture into {model_directory}'.format(
                      model_directory='data/models/' + model_name))
        os.mkdir(path + '/' + model_name)
        # Saving architecture
        yaml_string = model.to_yaml()
        with open(path + '/' + model_name + '/architecture.yml',
                  'w') as outfile:
            outfile.write(yaml_string)
        # Saving global parameters
        with open(path + '/' + model_name + '/initial_parameters.yml',
                  'w') as outfile:
            yaml.dump(initial_parameters, outfile, default_flow_style=False)
        # Saving estimator
        model.save(path + '/' + model_name + '/model.h5')


def save_model_performance(username, history, initial_parameters):
    file_path = os.path.dirname(os.path.abspath(__file__))
    path = file_path + '/../data/models'
    train_accuracy = history.history['acc']
    validation_accuracy = history.history['val_acc']

    train_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    list_epochs = np.arange(1, initial_parameters['epochs'] + 1)
    list_epochs = [str(epoch) for epoch in list_epochs]
    rows = zip(list_epochs, train_accuracy, validation_accuracy,
               train_loss, validation_loss)
    headers = ['Epoch', 'Train Accuracy', 'Validation Accuracy',
               'Train Loss', 'Validation Loss']
    model_names = [name for name in os.listdir(path + '/../data/models')
                   if name.startswith(username)]

    if len(model_names) == 0:
        model_name = username + '_' + '1'
        logging.info('Saving: model performance into {model_directory}'.format(
            model_directory=path + '/' + model_name))
        with open(path + '/' + model_name + '/evaluation.csv',
                  'w') as outfile:
            writer = csv.writer(outfile, delimiter='|')
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
    else:
        model_name = username + '_' + str(int(model_names[-1].split('_')[-1])
                                          + 1)
        logging.info('Saving: model performance into {model_directory}'.format(
            model_directory=path + '/' + model_name))
        with open(path + '/' + model_name + '/evaluation.csv',
                  'w') as outfile:
            writer = csv.writer(outfile, delimiter='|')
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
