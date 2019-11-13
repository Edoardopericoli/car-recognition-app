from Car_Prediction.models import Prototype
from Car_Prediction import pipeline


def test_run():
    pipeline.run(net=Prototype)
