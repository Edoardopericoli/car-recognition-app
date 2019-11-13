#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:54:59 2019

@author: valentinaparietti

Launch from terminal when you are in the folder containing the images of the cars you want to cut
"""

from imageai.Detection import ObjectDetection
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel()
images=["08145.jpg", "08146.jpg", "08147.jpg", "08214.jpg"]
for i in images:
    detections = detector.detectObjectsFromImage(input_image=i, output_image_path="new_{}".format(i), extract_detected_objects=True)
#    print(detections)

