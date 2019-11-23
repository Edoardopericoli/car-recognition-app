from CarModelClassifier.models import Prototype
from CarModelClassifier import pipeline
import os


def test_run():
    file_path = os.path.dirname(os.path.abspath(__file__))
    initial_parameters_path = file_path + "/../config/initial_parameters_test.yml"
    pipeline.run(initial_parameters_path=initial_parameters_path,
                 net=Prototype)
