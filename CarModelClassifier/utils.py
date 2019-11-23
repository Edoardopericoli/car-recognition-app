"""
Utils module.

List of functions
-----------------
"""
import logging
import numpy as np
import pandas as pd
import yaml
import os
import csv
from keras import backend as K
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from math import ceil


def setting_log():
    """
    Set basic log configuration.
    """
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def bound_cpu(n_threads=8):
    """
    Limit the CPU usage to a limited number of threads.
    
    Parameters
    ----------
    n_threads : int, optional
        number of threads, by default 8
    """
    K.set_session(K.tf.Session(
                   config=K.tf.ConfigProto
                   (intra_op_parallelism_threads=n_threads,
                    inter_op_parallelism_threads=n_threads)))


def load_parameters(parameters_path):
    """
    Load the parameters from the config file.
    
    Parameters
    ----------
    parameters_path : string
        path of the config file
    
    Returns
    -------
    dict
        a dict containing parameters.
    """
    with open(parameters_path) as f:
        initial_parameters = yaml.load(f)
    return initial_parameters


def load_labels_dfs(initial_parameters):
    """
    Load the labels files for training.
    
    Parameters
    ----------
    initial_parameters : dict
        a dict containing parameters.
    
    Returns
    -------
    Dataframes
        train and test dataframes.
    """
    file_path = Path((os.path.dirname(os.path.abspath(__file__))).replace('\\', '/'))
    train_df = pd.read_csv(file_path / ".." / initial_parameters['data_path'] / "labels/train_labels.csv")
    validation_df = pd.read_csv(file_path / ".." / initial_parameters['data_path'] / "labels/validation_labels.csv")
    train_df.reset_index(inplace=True)
    validation_df.reset_index(inplace=True)
    target_variable = 'model_label'
    train_df[target_variable] = train_df[target_variable].astype('str')
    validation_df[target_variable] = validation_df[target_variable].astype('str')
    return train_df, validation_df


def get_image_generators():
    """
    Create image generators for train and validation.
    
    Returns
    -------
    ImageDataGenerators
        ImageDataGenerator for train and validation.
    """
    train_image_generator = ImageDataGenerator(
                            rescale=1. / 255,
                            zoom_range=0.2,
                            rotation_range=5,
                            horizontal_flip=True)
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)
    return train_image_generator, validation_image_generator


def get_generator(imagedatagenerator, labels_df, directory,
                  initial_parameters, train=True):
    """
    Build the generator from dataframe.

    Parameters
    ----------
    imagedatagenerator : ImageDataGenerator
        ImageDataGenerator for the generator
    labels_df : DataFrame
        DataFrame containing labels.
    directory : string
        directory containing the images
    initial_parameters : dict
         a dict containing parameters.
    train : bool, optional
        if True is used for the training
        dataset, by default True

    Returns
    -------
    TrainGenerator
        the Train Generator.
    """
    if train:
        batch_size = initial_parameters['train_batch_size']
    else:
        batch_size = initial_parameters['validation_batch_size']

    file_path = Path((os.path.dirname(os.path.abspath(__file__))).replace('\\', '/'))
    directory = file_path / directory
    generator = imagedatagenerator.flow_from_dataframe(
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
    return generator


def train_model(train_generator, validation_generator, initial_parameters,
                train_df, model):
    """
    Perform the model training

    Parameters
    ----------
    train_generator : Generator
        Train Generator
    validation_generator : Generator
        Validation Generator
    initial_parameters : dict
        a dict containing parameters
    train_df : DataFrame
        train DataFrame
    model : Net
        Model used for training

    Returns
    -------
    DataFrame
        history DataFrame
    """
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=ceil(len(train_df) /
                                                       initial_parameters
                                                       ['train_batch_size']),
                                  validation_steps=ceil(len(train_df) /
                                                        initial_parameters
                                                        ['validation_batch_size']
                                                        ),
                                  validation_data=validation_generator,
                                  epochs=initial_parameters['epochs'],
                                  workers=8,
                                  max_queue_size=32,
                                  verbose=1)
    return history


def save_model_architecture(username, model, initial_parameters):
    """
    Save the entire model in a .h5 file.

    Save the model in a h5 file and the architecture in
    a yml file
    
    Parameters
    ----------
    username : string
        name of the folder in which put the saved the model
    model : Net
        the model used for training.
    initial_parameters : dict
        a dict containing parameters
    """
    file_path = Path((os.path.dirname(os.path.abspath(__file__))).replace('\\','/'))
    path = file_path / '..' / initial_parameters['data_path'] / 'models'
    if not path.is_dir():
        path.mkdir()
    model_names = [str(name) for name in path.glob('**/' + username + '*')]
    model_names = [int(i.split('_')[-1]) for i in model_names]

    if len(model_names) == 0:
        model_name = username + '_' + '1'
        logging.info('Saving: model, initial parameters \
                      and architecture into {model_directory}'.format(
                      model_directory=path / model_name))
        path = path / model_name
        path.mkdir()
        yaml_string = model.to_yaml()
        with open(path / 'architecture.yml',
                  'w') as outfile:
            outfile.write(yaml_string)
        # Saving global parameters
        with open(path / 'initial_parameters.yml',
                  'w') as outfile:
            yaml.dump(initial_parameters, outfile, default_flow_style=False)
        # Saving estimator
        model.save(str(path / 'model.h5'))
    else:
        model_name = username + '_' + str(max(model_names)+1)

        logging.info('Saving: model, initial parameters and \
                      architecture into {model_directory}'.format(
                      model_directory=path / model_name))
        path = path / model_name
        path.mkdir()
        # Saving architecture
        yaml_string = model.to_yaml()
        with open(path / 'architecture.yml',
                  'w') as outfile:
            outfile.write(yaml_string)
        # Saving global parameters
        with open(path / 'initial_parameters.yml',
                  'w') as outfile:
            yaml.dump(initial_parameters, outfile, default_flow_style=False)
        # Saving estimator
        model.save(str(path / 'model.h5'))


def save_model_performance(username, history, initial_parameters):
    """
    Save model performance in a DataFrame.

    Parameters
    ----------
    username : string
        name of the folder in which put the saved the model
    history : DataFrame
        History DataFrame
    initial_parameters : dict
        a dict containing parameters
    """
    file_path = Path((os.path.dirname(os.path.abspath(__file__))).replace('\\', '/'))
    path = file_path / '..' / initial_parameters['data_path'] / 'models'
    model_names = [str(name) for name in path.glob('**/' + username + '*')]
    model_names = [int(i.split('_')[-1]) for i in model_names]
    train_accuracy = history.history[list(history.history.keys())[0]]
    validation_accuracy = history.history[list(history.history.keys())[1]]

    train_loss = history.history[list(history.history.keys())[2]]
    validation_loss = history.history[list(history.history.keys())[3]]

    list_epochs = np.arange(1, initial_parameters['epochs'] + 1)
    list_epochs = [str(epoch) for epoch in list_epochs]
    rows = zip(list_epochs, train_accuracy, validation_accuracy,
               train_loss, validation_loss)
    headers = ['Epoch'] + list(history.history.keys())

    model_name = username + '_' + str(max(model_names))
    logging.info('Saving: model performance into {model_directory}'.format(
        model_directory=path / model_name))
    with open(path / model_name / 'evaluation.csv',
              'w') as outfile:
        writer = csv.writer(outfile, delimiter='|')
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def save_model_info(username, model, initial_parameters, history):
    """
    Save model architecture and performance
    
    Parameters
    ----------
    username : string
        name of the folder in which put the saved the model
    model : Net
        the model used for training.
    initial_parameters : dict
        a dict containing parameters
    history : DataFrame
        History DataFrame
    """
    save_model_architecture(username, model, initial_parameters)
    save_model_performance(username, history, initial_parameters)


def swish(x):
    """
    Custom activation function.
    
    Parameters
    ----------
    x : float
        input float
    
    Returns
    -------
    float
        the input number multiplied by the sigmoid function apply to it
    """
    return K.sigmoid(x)*x


class FixedDropout(layers.Dropout):
    """
    Fixed Dropout Layer
    """
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
