import pandas as pd
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
data_dict = loadmat(r'C:\Users\Edoardo\Desktop\devkit\cars_train_annos.mat')
data_dict2 = loadmat(r'C:\Users\Edoardo\Desktop\devkit\cars_meta.mat')

data_array = data_dict['annotations']
data_array = data_array.transpose(1, 0)

data_array2 = data_dict2['class_names']
data_array2 = data_array2.transpose(1, 0)

data2 = [data_array2[i][0][0].split(' ') for i in range(len(data_array2))]

brand = []
model = []
year = []
for labels in data2:
    brand.append(labels[0])
    model.append(' '.join(labels[1:-1]))
    year.append(labels[-1])

labels = pd.DataFrame({'label':np.arange(1,len(data2)+1), 'brand':brand, 'model':model, 'year': year})

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


data = pd.DataFrame({'fname':l6,'bbox_x1':l1,'bbbox_y1':l2,'bbox_x2':l3,'bbox_y2':l4,'class':l5,})

index = np.random.randint(0,len(data))
fname = data.loc[index,'fname']
im = np.array(Image.open(r'C:\Users\Edoardo\PycharmProjects\Car_Prediction\data\cars_train\{fname}'.format(fname=fname)), dtype=np.uint8)
fig, ax = plt.subplots(1)
ax.imshow(im)
rect = patches.Rectangle((data.iloc[index,1],data.iloc[index,2]),data.iloc[index,3]-data.iloc[index,1],data.iloc[index,4]-data.iloc[index,2],linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.show()