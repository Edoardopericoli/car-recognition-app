import pandas as pd
from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path

def prepare():
    # Getting data from the mat files
    file_path = Path((os.path.dirname(os.path.abspath(__file__)) + '/..').replace('\\','/'))
    print(file_path)
    data_dict = loadmat(file_path / 'data/raw_data/devkit/cars_train_annos.mat')
    data_dict2 = loadmat(file_path / 'data/raw_data/devkit/cars_meta.mat')

    data_array = data_dict['annotations']
    data_array = data_array.transpose(1, 0)

    data_array2 = data_dict2['class_names']
    data_array2 = data_array2.transpose(1, 0)

    # Getting a list of separate words for each observation
    data2 = [data_array2[i][0][0].split(' ') for i in range(len(data_array2))]

    # Extrapolating brand, model and year from the previous list
    brand = []
    model = []
    year = []
    for labels in data2:
        brand.append(labels[0])
        model.append(' '.join(labels[1:-1]))
        year.append(labels[-1])

    # Creating a DataFrame with brand, model and year for each label names (from 1 to 196)
    models = pd.DataFrame({'model_label': np.arange(1, len(data2)+1), 'brand': brand, 'model': model, 'year': year})
    brands_unique = models['brand'].unique()
    brands = pd.DataFrame({'brand_label': np.arange(1, len(brands_unique)+1), 'brand': brands_unique})

    # Extrapolating and transferring information about: bounding-boxes corners, name of photos and label names into a df
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    l6 = []
    for row in data_array:
        l1.append(row[0][0][0][0])
        l2.append(row[0][1][0][0])
        l3.append(row[0][2][0][0])
        l4.append(row[0][3][0][0])
        l5.append(row[0][4][0][0])
        l6.append(row[0][5][0])

    data = pd.DataFrame({'fname': l6, 'bbox_x1': l1, 'bbox_y1': l2, 'bbox_x2': l3, 'bbox_y2': l4, 'model_label': l5})
    data_temp = data.merge(models, left_on='model_label', right_on='model_label')
    data_temp = data_temp.merge(brands, left_on='brand', right_on='brand')
    data_final = data_temp[['fname', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'model_label', 'brand_label']]
    data = data_final.sort_values('fname')

    models.set_index('model_label', inplace=True)
    brands.set_index('brand_label', inplace=True)
    data.set_index('fname', inplace=True)

    if not os.path.exists(file_path / 'data/labels'):
        os.makedirs(file_path / 'data/labels')

    data.to_csv(file_path / 'data/labels/all_labels.csv')
    models.to_csv(file_path / 'data/labels/models_info.csv')
    brands.to_csv(file_path / 'data/labels/brands_info.csv')
