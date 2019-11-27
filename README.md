# CarModelClassifier

## Getting Started
The first part of the project allows to predict the model, the brand and the year of a car based
on an image. The second part of the project is about generating new car images using an alternative generator architecture (StyleGAN) for GANs. It is organized in 3 main files to be run by the user, 5 folders
for the code and 1 folder for the data. All the detailed info about the code and the functions
can be found in the _docs/docs.pdf_ file. The results obtained can be visualized on the _Report\_Cars_ Jupyter Notebook. 

## Installation
To install all the requirements run the following command:

```
pip install -r requirements.txt
```

To install the package go to the root directory and run the following command:

```
pip install .
```

## Prerequisites
The main requirement to satisfy is to download the _data_ folder from this link:   
and put in inside the main folder.
The downloaded folder contains the following data:
1.  __raw_data__:
    *  __train_cars_new__: All the train images must be in this folder
    * __YOLO_weights__: This contains the weights for object detection
    * __StyleGAN__: This contains the train images to generate car images with StyleGAN
2.  __labels__:
    *   __all_labels_new.csv__: This file should contain for each image name the real class
    * __models_info_new.csv__: This file combines each class with the model features, that is:
    _model_, _brand_ and _year_.
3. __models__: All the models already trained in a h5 format with all the parameters used for training. This folder is optional since a new train
    will update its content.
4. __train__: Train images used for the last training. This folder is optional since a new train
    will update its content.
5. __validation__: Validation images used for the last training. This folder is optional since a new train
    will update its content.
6. __test__: Test images. Useful for final evaluation of the model. This folder is optional since a new train
    will update its content.


## Project Organization
The code is organized in the following 5 folders:
1.  __CarModelClassifier__: This folder contains all the modules and the related 
functions to perform the main tasks of the project.
    *   __estimation.py__: In this file there are the functions for the model prediction and evaluation.
    *   __models.py__: This file contains an abstract class which identifies how the class of each new model
    should be set. It also contains the implementation of the models used in this project. Moreover it is possible to define custom models by implementing the Net interface.
    *   __pipeline.py__: This file contains the main function _run_ which defines the flow of the train.
    *   __splitter.py__: This file contains the function for the splitting of the data in train, test and validation.
    *   __utils.py__: Here are all the functions required in the different scripts.
    *   __yolo.py__: This script is necessary in the case an object location of the car and a subsequent
    cropping of the car is required.
2.  __config__: This folder contains the configuration files necessary to modify the main global parameters
for each model. Having configuration files allows not to modify the scripts directly every time it is necessary
to change some global parameters like the number of epochs or the size of the images.
3.   __docs__: This folder contains all the useful files necessary to create the documentation of the code
4.  __guess_make__: Here you can find all the material needed to create a graphical interface for the model
prediction.
5.  __tests__: Here are the tests of some functions of the project.


### Flow of training explanation
Here the main process of the project is explained in details. The steps are the following:
1.  Once the parameters described in [Training section](#Training) are defined, the file __train_main.py__ calls the function _run_ 
in the module __pipeline.py__. 
2.  Checking whether to crop images using the script __yolo.py__. If True, the function _crop_ gets
the yolo weights from the folder raw_data and performs object detection on the images. Then it crops
the images and put them in a folder called _object_detection_data_ inside the data folder.
3.  Checking whether to split the data using the script __splitter.py__. If True the splitting is performed
using the function _split_. If the parameter __crop_images__ is True, then the split is performed
using the images in the folder _data/object_detection_data/output_images_cropped_, otherwise using the
folder _data/raw_data/cars_train_new_. The splitting is performed in the following way:
    1.  The splitting is performed in a stratified fashion using the file _data/labels/all_labels_new.csv_.
    From this file 3 new csv files are created in the same folder:
     _train_labels.csv_, _test_labels.csv_, _validation_labels.csv_. Then, according to these 3 files
     the images in the initial folder are splitted in 3 new folders inside the __data__ folders:
     __train__, __test__, __validation__. 
4.  Checking whether to limit the cpu usage.
5.  Initializing the images using the keras generators.
6.  Initializing and training the pre-specified model.
7.  Saving the results for feature prediction, evaluation and to have an History of the models run.
The function used to save the results uses the _username_ pre-specified to create a folder having 
the same name inside the folder _data/models_. The name of the new created folder
will be a combination of the username and number, so that if you have more models having the 
same username, they will be saved with the same name and a chronological number at the end
to avoid the risk of overwriting existing folders. Inside this folder the function creates 4 files:
    *   __model.h5__: These are the weights after the training of the model. 
    *   __architecture.yml__: This is the architecture of the model.
    *   __evaluation.csv__: This is a csv file containing the loss and the accuracy in train and in validation
    for each epoch. This file is essential to crate plots from which one can see at which epoch the model starts
         overfitting. 
    *   __initial_parameters.yml__: This is the configuration file passed as a parameter.
        This allows the user to understand which were the values of the global parameters for each model.  

## Usage
### Training

To perform the train of the model run the following command:

```
python3 train_main.py
```

Default setting:

```
python3 train_main.py --username <trial> --params_file <initial_parameters.yml> --net <effnetb7> --shows_only_summary <False> --bounding_cpu <False> --split_data <True> --crop_images <False>
```

1.  Make sure you have satisfied the prerequisites.
2.  Launch from a terminal the script __train_main.py__ specifying the following parameters:
    *   __params_file__: This parameter lets the user specify the name of the configuration file,
    that must be placed in the folder __config__. The file to be passed must be a yaml file containing
    the following parameters:
        *   __seed__
        *   __train_batch_size__
        *   __validation_batch_size__
        *   __test_batch_size__
        *   __epochs__
        *   __IMG_HEIGHT__
        *   __IMG_WIDTH__
        *   __data_path__
    *   __username__: This will be the name of the folder in which important data, like the 
     model weights and architecture, will be saved for future usage.
    *   __shows_only_summary__: This is a boolean parameter which allows the user to decide whether to train the model or
    just to show the summary of the architecture to see the number of the parameters that will be trained. If True, the
    script stops right after having shown the model summary.
    *   __net__: This allows the user to specify which model architectures among the ones 
    present in _CarModelClassifier/models.py_. The possible values are:
        *   __effnetb1__
        *   __effnetb7__
        *   __prototype__
    *   __bounding_cpu__: A boolean parameter needed if the user wants to limit the cpu usage 
     for the process of training.
    *   __split_data__: This parameter allows the user to decide whether to perform the splitting of the data
    *   __crop_images__: If True the model performs object detection on the images, first locating the car
    and then cropping the image in such a way the new image will be only the box containing the car.

### Evaluation
To perform the evaluation of the model, that is to evaluate the accuracy on test images:

```
python3 evaluation_main.py
```

While to evaluate the accuracy on new images:

```
python3 evaluation_main.py custom_images True
```

1.  Make sure to have the following data:
    *   The folder _data/models/final_model_ and inside the files: 
        *   __model.h5__ 
        *   __initial_parameters.yml__
    *   If evaluating on test_data:
        *   _data/test_ containing the images.
        *   _data/labels/test_labels.csv_ containing for each image name the related class.
    *   If evaluating on new_data:
        *   The folder _custom_evaluaton_ containing:
            *   _images_ containing the new images.
            *   _test_labels.csv_ containing for each new image name the related class
2.   Launch from a terminal the script __evaluation_main.py__ specifying the following parameters:
        *   __custom_images__: If True the evaluation will be performed using new images in the folder
        _custom_evaluation_, otherwise the test data will be used.
        *   __test__: This parameter is True only when performing a test with pytest. Hence it should be
        False when performing prediction on new images.

### Prediction
To perform the prediction on new images:

```
python3 prediction_main.py
```

1.  Make sure to have the following data:
    *   The folder _data/models/final_model_ and inside the files: 
        *   __model.h5__ 
        *   __initial_parameters.yml__
    *   The file _models_info_new.csv_ inside the folder _data/labels_.
    *   The folder _custom_evaluaton_ containing:
            *   _images_ containing the new images.
            *   _test_labels.csv_ containing for each new image name the related class
2.   Launch from a terminal the script __prediction_main.py__ specifying the following parameters:
        *   __test__: This parameter is True only when performing a test with pytest. Hence it should be
        False when performing prediction on new images.

## WebApp
We provided this project with a Web Application that runs locally for presentation purposes. The implementation is available on the _guess-make_ folder. To execute locally this application __go inside the _guess-make_ folder__ and run the following command:

```
python3 app.py -path static
```

This application will give you the possibility to guess the type and brand of a car and challenge our model. You can try to beat it but it will not be easy!

## Running the tests
To test the most important modules of the project run the following command:

```
pytest
```

It is also available a complete report of the profiling of the training on the prototype model. The file in the folder _profilin_ contain all the information and can be visualized with the following commands:

Memory:
```
vprof --input-file profiling/memory.json
```

Profiler:
```
vprof --input-file profiling/profiler.json
```

Heatmap:
```
vprof --input-file profiling/heatmap.json
```

## Authors
*   __Martina Cioffi__ - https://github.com/martinacioffi
*   __Edoardo Manieri__ - https://github.com/edoardomanieri
*   __Valentina Parietti__ - https://github.com/ValentinaParietti
*   __Edoardo Pericoli__ - https://github.com/Edoardopericoli

