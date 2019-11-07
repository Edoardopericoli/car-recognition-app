from Car_Prediction.estimation import evaluation, prediction
import glob
import os


def test_evaluation():
    file_path = os.path.dirname(os.path.abspath(__file__))
    execution_path = file_path + '/test_model'
    test_images_path = file_path + '/test_images/images'
    test_labels_path = file_path + '/test_images/data.csv'
    accuracy = evaluation(execution_path, test_images_path, test_labels_path)
    assert (accuracy >= 0 and accuracy <= 1)


def test_prediction():
    file_path = os.path.dirname(os.path.abspath(__file__))
    execution_path = file_path + '/test_model'
    images_path = file_path + '/test_images/images'
    labels_info_path = file_path + '/test_images/labels_info.csv'
    n_images = len(glob.glob(images_path + "/*"))
    output_df = prediction(execution_path, images_path, labels_info_path)
    assert len(output_df) == n_images
