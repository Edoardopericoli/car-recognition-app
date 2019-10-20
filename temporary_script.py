from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import pandas as pd 
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from math import ceil
import yaml
import os

cwd = '/kaggle/working'
os.chdir(cwd)
os.chdir('../input/kermany2018/oct2017/OCT2017 ')

#Global parameters
#TODO chanigng batch_size so that all the steps have the same batch size and separating train batch size and val batch size
tr_batch_size = 2319
val_batch_size = 484
epochs = 2
IMG_HEIGHT = 100
IMG_WIDTH = 100

global_parameters = {'train_batch_size': tr_batch_size,
                     'validation_batch_size': val_batch_size,
                     'epochs': epochs,
                     'IMG_HEIGHT': IMG_HEIGHT,
                     'IMG_WIDTH': IMG_WIDTH}

# Assigning data in directories to global variables
PATH = os.getcwd()
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'test')

train_NORMAL_dir = os.path.join(train_dir, 'NORMAL') 
train_DRUSEN_dir = os.path.join(train_dir, 'DRUSEN')
train_CNV_dir = os.path.join(train_dir, 'CNV') 
train_DME_dir = os.path.join(train_dir, 'DME') 
validation_NORMAL_dir = os.path.join(validation_dir, 'NORMAL') 
validation_DRUSEN_dir = os.path.join(validation_dir, 'DRUSEN')
validation_CNV_dir = os.path.join(validation_dir, 'CNV') 
validation_DME_dir = os.path.join(validation_dir, 'DME')

num_NORMAL_tr = len(os.listdir(train_NORMAL_dir))
num_DRUSEN_tr = len(os.listdir(train_DRUSEN_dir))
num_CNV_tr = len(os.listdir(train_CNV_dir))
num_DME_tr = len(os.listdir(train_DME_dir))

num_NORMAL_val = len(os.listdir(validation_NORMAL_dir))
num_DRUSEN_val = len(os.listdir(validation_DRUSEN_dir))
num_CNV_val = len(os.listdir(validation_CNV_dir))
num_DME_val = len(os.listdir(validation_DME_dir))
total_train = num_NORMAL_tr + num_DRUSEN_tr + num_CNV_tr + num_DME_tr
total_val = num_NORMAL_val + num_DRUSEN_val + num_CNV_val + num_DME_val


train_image_generator = ImageDataGenerator(rescale=1./255) 
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=tr_batch_size,
                                                           directory=train_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=val_batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

#Model
model = Sequential([
    Conv2D(1, 8, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D((8,8)),
    Conv2D(1, 8, padding='same', activation='relu'),
    MaxPooling2D((8,8)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


#history = model.fit_generator(
#    train_data_gen,
#    epochs=epochs,
#    validation_data=val_data_gen,
#    steps_per_epoch=ceil(83484 / tr_batch_size)
#)

#TODO: finishing model saving
os.mkdir(cwd + '/models')
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
#acc = history.history['acc']
#val_acc = history.history['val_accuracy']

#loss = history.history['loss']
#val_loss = history.history['val_loss']

#epochs_range = range(epochs)


#Visualizing results
#plt.figure(figsize=(8, 8))
#plt.subplot(1, 2, 1)
#plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.title('Training and Validation Accuracy')

#plt.subplot(1, 2, 2)
#plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.title('Training and Validation Loss')
#plt.show()
