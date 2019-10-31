import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

# TODO: Adding some global parameters (train, validation and test size)

# Reading data
data = pd.read_csv('data/labels/all_labels.csv')

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
data = data.merge(dummies, how='inner', left_index=True, right_index=True)

train = train.merge(dummies, how='inner', left_index=True, right_index=True)
validation = validation.merge(dummies, how='inner', left_index=True, right_index=True)
test = test.merge(dummies, how='inner', left_index=True, right_index=True)

train.drop('class', axis=1, inplace=True)
validation.drop('class', axis=1, inplace=True)
test.drop('class', axis=1, inplace=True)

# Testing that the split has been executed correctly
assert len(data) == len(train) + len(validation) + len(test)

# Writing boxes data and class names data into csv files and writing a csv for each of train, validation and test
train.to_csv('data/labels/train_labels.csv')
validation.to_csv('data/labels/validation_labels.csv')
test.to_csv('data/labels/test_labels.csv')

#TODO: mettere one-hot encoding delle classi nel main

# Sending images to train, validation and test folders
indexes = {'train': train.index, 'validation': validation.index, 'test': test.index}
src = 'data/raw_data/cars_train'

for index in indexes.keys():
    dest = 'data/{index}'.format(index=index)
    for file_name in indexes[index]:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)
