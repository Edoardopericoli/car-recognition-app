from keras.models import load_model
from cv2 import cv2
import numpy as np
import os
import glob
import click
import pandas as pd
import yaml



@click.command()
@click.option('--execution_path', default=r"./", help='config model path',
              type=str)
@click.option('--images_path', default=r"./",
              help='path of images (file or directory)', type=str)
def main(execution_path, images_path):
    # Loading paths and model information
    initial_parameters_path = execution_path + '/initial_parameters.yml'
    with open(initial_parameters_path) as f:
        initial_parameters = yaml.load(f)
    model_path = execution_path + '/model.h5'
    model = load_model(model_path)

    if os.path.isfile(images_path):
        img = cv2.imread(images_path)
        img = cv2.resize(img, (initial_parameters['IMG_HEIGHT'],
                               initial_parameters['IMG_WIDTH']))
        images = np.reshape(img, [1, initial_parameters['IMG_HEIGHT'],
                                  initial_parameters['IMG_WIDTH'], 3])

        filenames = [os.path.basename(images_path)]

    elif os.path.isdir(images_path):
        images_path = glob.glob(images_path + "/*")
        images = [cv2.imread(f) for f in images_path]
        images = np.array([cv2.resize(img,
                                      (initial_parameters['IMG_HEIGHT'],
                                       initial_parameters['IMG_WIDTH']))
                          for img in images])
        images = np.reshape(images, [len(images_path),
                                     initial_parameters['IMG_HEIGHT'],
                                     initial_parameters['IMG_WIDTH'], 3])

        filenames = [os.path.basename(img_path) for img_path in images_path]
    classes = model.predict_classes(images)

    output_df = pd.DataFrame({'filename': filenames, 'class': classes})
    print(output_df)


if __name__ == "__main__":
    main()