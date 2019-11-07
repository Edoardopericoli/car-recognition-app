from Car_Prediction.estimation import evaluation
import click


# /home/edoardo/Desktop/University/Cars_Classification/Car_Prediction/trial_model
# /home/edoardo/Desktop/University/Cars_Classification/Car_Prediction/trial_images
# /home/edoardo/Desktop/University/Cars_Classification/Car_Prediction/data/labels_info.csv
@click.command()
@click.option('--execution_path', default=r"./", help='config model path',
              type=str)
@click.option('--test_images_path', default=r"./",
              help='path of images (file or directory)', type=str)
@click.option('--test_labels_path', default=r"./",
              help='path of images (file or directory)', type=str)
def main(execution_path, test_images_path, test_labels_path):
    accuracy = evaluation(execution_path, test_images_path, test_labels_path)
    print(f'Accuracy: {accuracy:.2f}')
