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
    dirs = list(filter(os.path.isdir, [origin_images_path + '/' + i for i in os.listdir(origin_images_path)]))

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
    dirs = filter(os.path.isdir, [origin_images_path + '/' + i for i in os.listdir(origin_images_path)])

    for directory in dirs:
        files = os.listdir(directory)
        for file in files:
            number = re.search(r"[0-9]+", str(directory))
            new_name = str(number.group())+".jpg"
            start = directory + str("/") + file
            end = directory + str("/") + new_name
            os.rename(start, end)

    # Put the all the cut cars into a folder named "output_images_cropped"
    dirs = filter(os.path.isdir, [origin_images_path + '/' + i for i in os.listdir(origin_images_path)])
    if not os.path.exists(destination_images_path + "/output_images_cropped"):
        os.mkdir(destination_images_path + "/output_images_cropped")

    for directory in dirs:
        files = os.listdir(directory)
        for file in files:
            start = directory + str("/") + file
            destination = str(destination_images_path + "/output_images_cropped") + str("/") + file
            shutil.copyfile(start, destination)


if __name__ == "__main__":
    main()
