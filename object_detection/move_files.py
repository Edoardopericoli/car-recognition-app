#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:13:22 2019

@author: valentinaparietti

Launch from terminal when you are in the folder containing the folders with the objects found for each image
"""

from PIL import Image
import shutil
import os
#import time
import re

#keep only the biggest cut car 
dirs = filter(os.path.isdir, os.listdir(os.curdir))
for dir in dirs:
    files = os.listdir(str(os.curdir)+ str("/") + dir)
    cars_size = {}
    
    for file in files:
        if not file.startswith('car'):
#            print(f'\n The image named {file} in the folder {dir} does not start with -car- and so it will be removed \n')
#            time.sleep(5)
            os.unlink(str(os.curdir)+ str("/") + dir + str("/") +file)
    remaining_files = os.listdir(str(os.curdir)+ str("/") + dir)
    
    for file in remaining_files:
        cars_size[file] = Image.open(str(os.curdir)+ str("/") + dir + str("/") +file).size
    if len(cars_size) > 1:
#        print(f"In the folder {dir} there are multiple images: {list(cars_size.keys())}: \n")
        biggest_car = None
        dim = (0,0)
        
        for car in cars_size.keys():
            if cars_size[car][0] * cars_size[car][1] > dim[0] * dim[1]:
                biggest_car = car
                dim = cars_size[car]
#        print(f"The biggest image is: {biggest_car}: \n")
        to_delete = (list(set(cars_size.keys())))
        to_delete.remove(biggest_car)
#        print(f"The following images: {to_delete} will be removed \n\n")
#        time.sleep(5)
        
        for small_car in to_delete:
            os.unlink(str(os.curdir)+ str("/") + dir + str("/") +small_car)
            
#Rename the images as number.jpg    
dirs = filter(os.path.isdir, os.listdir(os.curdir))
for dir in dirs:    
    files = os.listdir(str(os.curdir)+ str("/") + dir) 
    for file in files:
        number = re.search(r"[0-9]+", str(dir))
        new_name=str(number.group())+".jpg"
        start = str(os.curdir)+ str("/") + dir + str("/") + file
        end = str(os.curdir)+ str("/") + dir + str("/") + new_name
        os.rename(start,end)
 
#Put the all the cut cars into a folder named "output images"       
dirs = filter(os.path.isdir, os.listdir(os.curdir))
os.mkdir("output_images")
for dir in dirs:    
    files = os.listdir(str(os.curdir)+ str("/") + dir) 
    for file in files:
        start = str(os.curdir)+ str("/") + dir + str("/") + file 
        destination = str(os.curdir) + str("/") + "output_images" + str("/") + file
        shutil.copyfile(start, destination)