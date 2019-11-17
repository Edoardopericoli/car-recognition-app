from imageai.Detection import ObjectDetection
from PIL import Image
import shutil
import os
import re
import click
import logging

# todo: to be transferred in the pipeline of the model
#  --> creating a script "cropper" with the function crop
#  inside the directory Car_Prediction


@click.command()
@click.option('--origin_images_path', default=r"../data/raw_data/cars_train", help='folder from which detecting images',
              type=str)
@click.option('--destination_images_path', default=r"../data/object_detection_data/output_images_YOLO",
              help='folder from which detecting images', type=str)
@click.option('--sample', default=False,
              help='if True detects only a sample of the images', type=bool)
def main(origin_images_path="../data/raw_data/cars_train",
         destination_images_path="../data/object_detection_data/output_images_YOLO",
         final_images_path="../data/object_detection_data", sample=True):

    if not os.path.exists(destination_images_path):
        os.makedirs(destination_images_path)

    logging.info('Starting detecting objects')
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("../data/raw_data/YOLO_weights/yolo.h5")
    detector.loadModel()
    images = os.listdir(origin_images_path)

    if sample:
        images = images[:10]

    for i in images:
        detector.detectObjectsFromImage(input_image=origin_images_path + '/' + i,
                                        output_image_path=destination_images_path + "/new_{}".format(i),
                                        extract_detected_objects=True)
    logging.info('Finished detecting objects')
    logging.info('Starting assigining objects to folder output_image_cropped')
   # Keep only the biggest cut car
    dirs = list(filter(os.path.isdir, [destination_images_path + '/' + i for i in os.listdir(destination_images_path)]))

    for directory in dirs:
        files = os.listdir(directory)
        cars_size = {}
        for file in files:
            if not (file.startswith('car') or file.startswith('truck')):
                os.unlink(directory + "/" + file)
        remaining_files = os.listdir(directory)

        for file in remaining_files:
            cars_size[file] = Image.open(directory + str("/") + file).size
        if len(cars_size) > 1:
            biggest_car = None
            dim = (0, 0)

            for car in cars_size.keys():
                if cars_size[car][0] * cars_size[car][1] > dim[0] * dim[1]:
                    biggest_car = car
                    dim = cars_size[car]

            to_delete = (list(set(cars_size.keys())))
            to_delete.remove(biggest_car)

            for small_car in to_delete:
                os.unlink(directory + str("/") + small_car)

    # Rename the images as number.jpg
    dirs = filter(os.path.isdir, [destination_images_path + '/' + i for i in os.listdir(destination_images_path)])

    for directory in dirs:
        files = os.listdir(directory)
        for file in files:
            number = re.search(r"[0-9]+", str(directory))
            new_name = str(number.group())+".jpg"
            start = directory + str("/") + file
            end = directory + str("/") + new_name
            os.rename(start, end)

    # Put the all the cut cars into a folder named "output_images_cropped"
    dirs = filter(os.path.isdir, [destination_images_path + '/' + i for i in os.listdir(destination_images_path)])
    if not os.path.exists(final_images_path + "/output_images_cropped"):
        os.mkdir(final_images_path + "/output_images_cropped")

    for directory in dirs:
        files = os.listdir(directory)
        for file in files:
            start = directory + str("/") + file
            destination = str(final_images_path + "/output_images_cropped") + str("/") + file
            shutil.copyfile(start, destination)

    logging.info('Finished entire process')
if __name__ == "__main__":
    main()
