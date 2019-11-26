"""
Estimation module.

List of functions
-----------------
"""
from keras.models import load_model
from cv2 import cv2
import numpy as np
import os
import glob
import pandas as pd
import yaml
from CarModelClassifier.utils import swish, FixedDropout


def evaluation(custom_images=False, test=False):
    """Perform evaluation of the model.

    Images to evaluate the model are taken from the folder
    "custom_evaluation/images" while the labels of these images
    are in the file "/custom_evaluation/test_labels.csv"

    Parameters
    ----------
    custom_images : bool, optional
        if True images to evaluate the model are taken from the folder
        "custom_evaluation/images" while the labels of these images
        are those in the file "/custom_evaluation/test_labels.csv".
        if False images to evaluate the model are taken from the folder
        "data/test" while the labels of these images
        are those in the file "data/labels/test_labels.csv".
    test : bool, optional
        if True, test the correct working of the function, 
        by default False
    
    Returns
    -------
    float
        accuracy of the model.
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    test_images_path = file_path + '/../data/test'
    execution_path = file_path + '/../data/models/final_model'
    test_labels_path = file_path + '/../data/labels/test_labels.csv'

    if custom_images:
        test_images_path = file_path + '/../custom_evaluation/images'
        execution_path = file_path + '/../data/models/final_model'
        test_labels_path = file_path + '/../custom_evaluation/test_labels.csv'
    if test:
        execution_path = file_path + '/../tests/test_model'
        test_images_path = file_path + '/../tests/test_images/images'
        test_labels_path = file_path + '/../tests/test_images/data.csv'
    # Loading paths and model information
    initial_parameters_path = execution_path + '/initial_parameters.yml'
    with open(initial_parameters_path) as f:
        initial_parameters = yaml.load(f)
    model_path = execution_path + '/model.h5'
    model = load_model(model_path, custom_objects={
                                                  'swish': swish,
                                                  'FixedDropout': FixedDropout
                                                  })

    test_images_path = glob.glob(test_images_path + "/*")
    images = [cv2.imread(f) for f in test_images_path]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    images = [cv2.resize(img, (initial_parameters['IMG_HEIGHT'],
                         initial_parameters['IMG_WIDTH']))
              for img in images]
    images = np.array([image.astype("float") / 255.0 for image in images])
    images = np.reshape(images, [len(test_images_path),
                                 initial_parameters['IMG_HEIGHT'],
                                 initial_parameters['IMG_WIDTH'], 3])
    filenames = [os.path.basename(img_path)
                 for img_path in test_images_path]

    classes_lists = model.predict(images)
    class_n = classes_lists.argmax(axis=1).tolist()

    labels = sorted([str(i + 1) for i in range(41)])
    classes = [int(labels[el]) for el in class_n]

    output_df = pd.DataFrame({'filename': filenames,
                              'predicted_class': classes})
    labels_df = pd.read_csv(test_labels_path, sep=',')
    output_df = output_df.merge(labels_df, left_on='filename',
                                right_on='fname')
    print(output_df.head(30))
    accuracy = len(output_df.loc[output_df['predicted_class'] ==
                   output_df['model_label'], :]) / len(output_df)
    return accuracy


def prediction(test=False):
    """Perform prediction of new images.

    Images to evaluate the model are taken from the folder
    "custom_evaluation/images".

    Parameters
    ----------
    test : bool, optional
        if True, used to test
        the function, by default False

    Returns
    -------
    DataFrame
        The output dataframe.
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    test_images_path = file_path + '/../custom_evaluation/images'
    execution_path = file_path + '/../data/models/final_model'
    labels_info_path = file_path + '/../data/labels/models_info_new.csv'
    if test:
        execution_path = file_path + '/../tests/test_model'
        test_images_path = file_path + '/../tests/test_images/images'
        labels_info_path = file_path + '/../tests/test_images/labels_info.csv'
    # Loading paths and model information
    initial_parameters_path = execution_path + '/initial_parameters.yml'
    with open(initial_parameters_path) as f:
        initial_parameters = yaml.load(f)
    model_path = execution_path + '/model.h5'
    model = load_model(model_path, custom_objects={
                                                  'swish': swish,
                                                  'FixedDropout': FixedDropout
                                                  })

    test_images_path = glob.glob(test_images_path + "/*")
    images = [cv2.imread(f) for f in test_images_path]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    images = [cv2.resize(img, (initial_parameters['IMG_HEIGHT'],
                         initial_parameters['IMG_WIDTH']))
              for img in images]
    images = np.array([image.astype("float") / 255.0 for image in images])
    images = np.reshape(images, [len(test_images_path),
                                 initial_parameters['IMG_HEIGHT'],
                                 initial_parameters['IMG_WIDTH'], 3])
    filenames = [os.path.basename(img_path)
                 for img_path in test_images_path]

    classes_lists = model.predict(images)
    if len(images) == 1:
        class_n = np.asarray(classes_lists).argmax().tolist()
    else:
        class_n = classes_lists.argmax(axis=1).tolist()
    labels = sorted([str(i + 1) for i in range(41)])
    classes = [int(labels[el]) for el in class_n]

    output_df = pd.DataFrame({'filename': filenames, 'class': classes})
    labels_df = pd.read_csv(labels_info_path)
    output_df = output_df.merge(labels_df, left_on='class', right_on='label') \
                         .drop(columns=['class'])
    return output_df

