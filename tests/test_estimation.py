from CarModelClassifier.estimation import evaluation, prediction
import glob
import os


def test_evaluation():
    accuracy = evaluation(test=True)
    assert (accuracy >= 0 and accuracy <= 1)


def test_prediction():
    file_path = os.path.dirname(os.path.abspath(__file__))
    images_path = file_path + '/test_images/images'
    n_images = len(glob.glob(images_path + "/*"))
    output_df = prediction(test=True)
    assert len(output_df) == n_images
