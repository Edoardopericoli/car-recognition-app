from imageai.Detection import ObjectDetection
import os
import click


@click.command()
@click.option('--origin_images_path', default=r"../data/raw_data/cars_train/", help='folder from which detecting images',
              type=str)
@click.option('--destination_images_path', default=r"../data/object_detection_data/output_images_YOLO/",
              help='folder from which detecting images', type=str)
@click.option('--sample', default=False,
              help='if True detects only a sample of the images', type=bool)
def main(origin_images_path, destination_images_path, sample):

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("../data/raw_data/YOLO_weights/yolo.h5")
    detector.loadModel()
    images = os.listdir(origin_images_path)

    if sample:
        images = images[:10]

    for i in images:
        detector.detectObjectsFromImage(input_image=origin_images_path + i,
                                        output_image_path=destination_images_path + "new_{}".format(i),
                                        extract_detected_objects=True)


if __name__ == "__main__":
    main()
