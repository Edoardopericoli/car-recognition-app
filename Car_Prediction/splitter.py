import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
from pathlib import Path
import logging


def split(train_size=0.8, target_variable='model', origin_data_path='/data/labels/all_labels.csv', get_cropped_data_stanford=False):
    file_path = Path((os.path.dirname(os.path.abspath(__file__)) + '/..').replace('\\','/'))
    assert target_variable in ['brand', 'model']

    # Reading data
    data = pd.read_csv(file_path / origin_data_path)

    if get_cropped_data_stanford:
        print('Data are no longer taken from ' + origin_data_path)
        print('If get_cropped_data_stanford is True data are taken from the path in which cropped images are located')
        data = pd.read_csv('../data/object_detection_data/output_images_cropped')

    if target_variable == 'brand':
        # Splitting train, validation, test
        X_train, X_test_temp, y_train, y_test_temp = train_test_split(data[['fname', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']],
                                                                      data['brand_label'],
                                                                      test_size=1-train_size,
                                                                      random_state=89,
                                                                      stratify=data['brand_label']
                                                                      )

        train = pd.DataFrame(X_train).merge(pd.DataFrame(y_train), left_index=True, right_index=True)
        test_temp = pd.DataFrame(X_test_temp).merge(pd.DataFrame(y_test_temp), left_index=True, right_index=True)

        X_test, X_validation, y_test, y_validation = train_test_split(test_temp[['fname', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']],
                                                                      test_temp['brand_label'],
                                                                      test_size=0.5,
                                                                      random_state=89,
                                                                      stratify=test_temp['brand_label']
                                                                      )

    elif target_variable == 'model':
        # Splitting train, validation, test
        X_train, X_test_temp, y_train, y_test_temp = train_test_split(data[['fname', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']],
                                                                      data['model_label'],
                                                                      test_size=1 - train_size,
                                                                      random_state=89,
                                                                      stratify=data['model_label']
                                                                      )

        train = pd.DataFrame(X_train).merge(pd.DataFrame(y_train), left_index=True, right_index=True)
        test_temp = pd.DataFrame(X_test_temp).merge(pd.DataFrame(y_test_temp), left_index=True, right_index=True)

        X_test, X_validation, y_test, y_validation = train_test_split(test_temp[['fname', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']],
                                                                      test_temp['model_label'],
                                                                      test_size=0.5,
                                                                      random_state=89,
                                                                      stratify=test_temp['model_label']
                                                                      )

    validation = pd.DataFrame(X_validation).merge(pd.DataFrame(y_validation), left_index=True, right_index=True)
    test = pd.DataFrame(X_test).merge(pd.DataFrame(y_test), left_index=True, right_index=True)

    data.set_index('fname', inplace=True)
    train.set_index('fname', inplace=True)
    validation.set_index('fname', inplace=True)
    test.set_index('fname', inplace=True)

    # Testing that the split has been executed correctly
    assert len(data) == len(train) + len(validation) + len(test)

    # Writing boxes data and class names data into csv files and writing a csv for each of train, validation and test
    train.to_csv(file_path / 'data/labels/train_labels.csv')
    validation.to_csv(file_path / 'data/labels/validation_labels.csv')
    test.to_csv(file_path / 'data/labels/test_labels.csv')

    # Sending images to train, validation and test folders
    indexes = {'train': train.index, 'validation': validation.index, 'test': test.index}
    src = file_path / 'data/raw_data/cars_train'

    for index in indexes.keys():
        dest = 'data/{index}'.format(index=index)
        if not os.path.exists(file_path / dest):
            os.makedirs(file_path / dest)
        for file_name in indexes[index]:
            full_file_name = src / file_name
            if full_file_name.is_file():
                shutil.copy(full_file_name, dest)
