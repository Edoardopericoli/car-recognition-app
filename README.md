# Car_Prediction

The process of the project is the following:
1.  _label_preparation.py_: this file transforms the format of the data 
from a mata file to csv and splits the data into train, validation and test
according to some sizes. Files and images are splitted and inserted in the "data" 
folder.
2.  _main.py_: this file performs the loading of data, the model creation
and model fitting of the train data. Finally aims to evaluate the model
using the validation data and saving the model, the architecture, the initial parameters
and the score into a subfolder inside the folder "models". Each of the
subfolder will be named in chronological order, so we can have a history 
of the models run. The main file can be executed using from UNIX Shell.

