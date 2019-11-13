from Car_Prediction.net import Net
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential


class Prototype(Net):

    def __init__(self, train_generator, initial_parameters=None):
        if initial_parameters is None:
            self.initial_parameters = {}
        else:
            self.initial_parameters = initial_parameters
        self.train_generator = train_generator
        self.model = self.setup_model()

    def setup_model(self):
        model = Sequential([
            Conv2D(5, 4, activation='relu', input_shape=(self.initial_parameters['IMG_HEIGHT'],
                                                         self.initial_parameters['IMG_WIDTH'], 3)),
            MaxPooling2D((8, 8)),
            Dropout(0.1, seed=self.initial_parameters['seed']),
            Conv2D(5, 4, padding='same', activation='relu'),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(196, activation='softmax')
            ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


