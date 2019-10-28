# Car_Prediction

The process of the project is the following:
1.  _label_preparation.py_: this file transforms the format of the data 
from a mat file to csv. 
2. _splitting_data.py_: this file splits the data in train, validation ad test sets.
3.  _main.py_: this file performs the loading of data, the model creation
and model fitting of the train data. Finally aims to evaluate the model
using the validation data and saving the model, the architecture, the initial parameters
and the score into a subfolder inside the folder "models". Each of the
subfolder will be named in chronological order, so we can have a history 
of the models run. The main file can be executed using UNIX Shell.

