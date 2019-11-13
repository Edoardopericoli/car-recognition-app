from keras.layers import Dense
from keras import optimizers
from keras.layers import GlobalAveragePooling2D, BatchNormalization
from keras import Model
import efficientnet.keras as efn
from Car_Prediction.net import Net


class Effnet(Net):

    def __init__(self, train_generator):
        self.train_generator = train_generator
        self.model = self.setup_model()

    def setup_model(self):
        base_model = self._setup_base_model()
        return self._setup_final_layers(base_model)

    def _setup_base_model(self):
        base_model = efn.EfficientNetB1(weights='imagenet', include_top=False)
        # fix the feature extraction part of the model
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        return base_model

    def _setup_final_layers(self, base_model):
        x = GlobalAveragePooling2D()(base_model.output)
        predictions = Dense(len(self.train_generator.class_indices),
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(lr=0.01),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        return model
