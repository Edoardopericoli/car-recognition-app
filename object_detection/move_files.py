from PIL import Image
import shutil
import os
import re
import click


@click.command()
@click.option('--origin_images_path', default=r"../data/object_detection_data/output_images_YOLO", help='folder from which detecting images',
              type=str)
@click.option('--destination_images_path', default=r"../data/object_detection_data",
              help='folder from which detecting images', type=str)
def main(origin_images_path, destination_images_path):
    # Keep only the biggest cut car
    dirs = filter(os.path.isdir, os.listdir(origin_images_path))

    for directory in dirs:
        files = os.listdir(str(origin_images_path) + str("/") + directory)
        cars_size = {}

        for file in files:
            if not file.startswith('car'):
                os.unlink(str(origin_images_path) + str("/") + directory + str("/") + file)
        remaining_files = os.listdir(str(origin_images_path) + str("/") + directory)

        for file in remaining_files:
            cars_size[file] = Image.open(str(origin_images_path) + str("/") + directory + str("/") + file).size
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
                os.unlink(str(origin_images_path) + str("/") + directory + str("/") + small_car)

    # Rename the images as number.jpg
    dirs = filter(os.path.isdir, os.listdir(origin_images_path))

    for directory in dirs:
        files = os.listdir(str(origin_images_path) + str("/") + directory)
        for file in files:
            number = re.search(r"[0-9]+", str(directory))
            new_name = str(number.group())+".jpg"
            start = str(origin_images_path) + str("/") + directory + str("/") + file
            end = str(origin_images_path) + str("/") + directory + str("/") + new_name
            os.rename(start, end)

    # Put the all the cut cars into a folder named "output_images_chopped"
    dirs = filter(os.path.isdir, os.listdir(destination_images_path))
    if not os.path.exists(destination_images_path + "/output_images_chopped"):
        os.makedirs(destination_images_path + "/output_images_chopped")

    for directory in dirs:
        files = os.listdir(str(origin_images_path) + str("/") + directory)
        for file in files:
            start = str(origin_images_path) + str("/") + directory + str("/") + file
            destination = str(destination_images_path + "/output_images_chopped") + str("/") + file
            shutil.copyfile(start, destination)


if __name__ == "__main__":
    main()
