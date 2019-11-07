# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


from keras.models import load_model

# load model
model = load_model('data/models/pericoli_1/model.h5')
model.predict_classes()
