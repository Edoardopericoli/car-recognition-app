import pandas as pd
from scipy.io import loadmat
import numpy as np

# Getting data from the mat files
data_dict = loadmat(r'data\raw_data\devkit\cars_train_annos.mat')
data_dict2 = loadmat(r'data\raw_data\devkit\cars_meta.mat')

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

# Creating a DataFrame with brand, model and year for each class names (from 1 to 196)
labels = pd.DataFrame({'label': np.arange(1,len(data2)+1), 'brand': brand, 'model': model, 'year': year})
labels.set_index('label', inplace=True)

# Extrapolating and transferring information about: bounding-boxes corners, name of photos and class names into a df
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

# Getting 196 dummies for the classes because Keras reads data in this format
data = pd.DataFrame({'fname': l6, 'bbox_x1': l1, 'bbox_y1': l2, 'bbox_x2': l3, 'bbox_y2': l4, 'class': l5})

data.to_csv('data/labels/all_labels.csv')
labels.to_csv('data/labels/labels_info.csv')



