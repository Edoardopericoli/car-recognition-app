import numpy as np
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation
from keras import optimizers
import tensorflow as tf
import csv
import click
import logging
import os
import yaml
import sys
from keras.layers import GlobalAveragePooling2D, BatchNormalization
from keras import Model
import efficientnet.keras as efn

@click.command()
@click.option('--initial_parameters_path', default=r"../config/initial_parameters.yml", help='config file containing initial parameters', type=str)
@click.option('--username', help='username to be used for model saving', type=str)
@click.option('--shows_only_summary', default=False, help='if True the program stops after having shown the model summary', type=bool)
def main(initial_parameters_path, username, shows_only_summary):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.info('Starting the process')
    logging.info('Asserting dimensions of train, validation and test')

    batch_size = 64
    input_shape = (240, 240)
    data_dir = '../car_data/car_data/'
    train_dir = data_dir + 'train'
    test_dir = data_dir + 'test'

    logging.info('Transforming data using ImageDataGenerator')
    train_datagen = ImageDataGenerator(
                            rescale=1. / 255,
                            zoom_range=0.2,
                            rotation_range=5,
                            horizontal_flip=True)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_generator=train_datagen.flow_from_directory(train_dir,
                                            class_mode="categorical", 
                                            target_size=input_shape, 
                                            batch_size=batch_size)


    validation_generator=test_datagen.flow_from_directory(test_dir,
                                            class_mode="categorical", 
                                            target_size=input_shape, 
                                            batch_size=batch_size)

    base_model = efn.EfficientNetB1(weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # fix the feature extraction part of the model
    for layer in base_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
        
    model.compile(optimizer=optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.samples // batch_size + 1 ,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size + 1,
                    epochs=10,                           
                    workers=8,             
                    max_queue_size=32,             
                    verbose=1)


    if shows_only_summary:
        sys.exit()


    # Saving model
    model_names = [name for name in os.listdir('../data/models') if name.startswith(username)]
    if len(model_names) == 0:
        model_name = username + '_' + '1'
        logging.info('Saving: model, initial parameters and architecture into {model_directory}'.format(
            model_directory='../data/models/' + model_name))
        os.mkdir('../data/models/' + model_name)
        yaml_string = model.to_yaml()
        with open('../data/models/' + model_name + '/architecture.yml', 'w') as outfile:
            outfile.write(yaml_string)
        # Saving global parameters
        with open('../data/models/' + model_name + '/initial_parameters.yml', 'w') as outfile:
            yaml.dump(initial_parameters, outfile, default_flow_style=False)
        # Saving estimator
        model.save('../data/models/' + model_name + '/model.h5')
    else:
        model_name = username + '_' + str(int(model_names[-1].split('_')[-1]) + 1)
        logging.info('Saving: model, initial parameters and architecture into {model_directory}'.format(
            model_directory='data/models/' + model_name))
        os.mkdir('../data/models/' + model_name)
        # Saving architecture
        yaml_string = model.to_yaml()
        with open('../data/models/' + model_name + '/architecture.yml', 'w') as outfile:
            outfile.write(yaml_string)
        # Saving global parameters
        with open('../data/models/' + model_name + '/initial_parameters.yml', 'w') as outfile:
            yaml.dump(initial_parameters, outfile, default_flow_style=False)
        # Saving estimator
        model.save('../data/models/' + model_name + '/model.h5')

    # Model Performance
    train_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    train_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    list_epochs = np.arange(1, initial_parameters['epochs'] + 1)
    list_epochs = [str(epoch) for epoch in list_epochs]
    rows = zip(list_epochs, train_accuracy, validation_accuracy, train_loss, validation_loss)
    headers = ['Epoch', 'Train Accuracy', 'Validation Accuracy', 'Train Loss', 'Validation Loss']
    if len(model_names) == 0:
        model_name = username + '_' + '1'
        logging.info('Saving: model performance into {model_directory}'.format(
            model_directory='../data/models/' + model_name))
        with open('../data/models/' + model_name + '/evaluation.csv', 'w') as outfile:
            writer = csv.writer(outfile, delimiter='|')
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
    else:
        model_name = username + '_' + str(int(model_names[-1].split('_')[-1]) + 1)
        logging.info('Saving: model performance into {model_directory}'.format(
            model_directory='../data/models/' + model_name))
        with open('../data/models/' + model_name + '/evaluation.csv', 'w') as outfile:
            writer = csv.writer(outfile, delimiter='|')
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)

    logging.info('Process finished')


if __name__ == '__main__':
    main()