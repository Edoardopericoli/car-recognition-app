import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import shutil

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

# TODO: Separare label preparation (from mat to csv) dall'ingestion (splitting train, validation, test)
# TODO: Adding some global parameters (train, validation and test size)
# Splitting train, validation, test
X_train, X_test_temp, y_train, y_test_temp = train_test_split(data[['fname', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']],
                                                              data['class'],
                                                              test_size=0.2,
                                                              random_state=89,
                                                              stratify=data['class']
                                                              )

train = pd.DataFrame(X_train).merge(pd.DataFrame(y_train), left_index=True, right_index=True)
test_temp = pd.DataFrame(X_test_temp).merge(pd.DataFrame(y_test_temp), left_index=True, right_index=True)

X_test, X_validation, y_test, y_validation = train_test_split(test_temp[['fname', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']],
                                                              test_temp['class'],
                                                              test_size=0.5,
                                                              random_state=89,
                                                              stratify=test_temp['class']
                                                              )

validation = pd.DataFrame(X_validation).merge(pd.DataFrame(y_validation), left_index=True, right_index=True)
test = pd.DataFrame(X_test).merge(pd.DataFrame(y_test), left_index=True, right_index=True)

data.set_index('fname', inplace=True)
train.set_index('fname', inplace=True)
validation.set_index('fname', inplace=True)
test.set_index('fname', inplace=True)

dummies = pd.get_dummies(data['class'])
train = train.merge(dummies, how='inner', left_index=True, right_index=True)
validation = validation.merge(dummies, how='inner', left_index=True, right_index=True)
test = test.merge(dummies, how='inner', left_index=True, right_index=True)

train.drop('class', axis=1, inplace=True)
validation.drop('class', axis=1, inplace=True)
test.drop('class', axis=1, inplace=True)

# Writing boxes data and class names data into csv files and writing a csv for each of train, validation and split
train.to_csv('data/labels/train_labels.csv')
validation.to_csv('data/labels/validation_labels.csv')
test.to_csv('data/labels/test_labels.csv')
labels.to_csv('data/labels/labels_info.csv')

# Sending images to train, validation and test folders
indexes = {'train':train.index, 'validation':validation.index, 'test':test.index}
src = 'data/raw_data/cars_train'

for index in indexes.keys():
    dest = 'data/{index}'.format(index=index)
    for file_name in indexes[index]:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)



# Code for visualization
# index = np.random.randint(0,len(data))
# fname = data.loc[index,'fname']
# im = np.array(Image.open(r'C:\Users\Edoardo\PycharmProjects\Car_Prediction\data\cars_train\{fname}'.format(fname=fname)), dtype=np.uint8)
# fig, ax = plt.subplots(1)
# ax.imshow(im)
# rect = patches.Rectangle((data.iloc[index,1],data.iloc[index,2]),data.iloc[index,3]-data.iloc[index,1],data.iloc[index,4]-data.iloc[index,2],linewidth=1,edgecolor='r',facecolor='none')
# ax.add_patch(rect)
# plt.show()