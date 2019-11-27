from keras import optimizers
from keras.layers import GlobalAveragePooling2D, BatchNormalization
from keras import Model
import efficientnet.keras as efn
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential
import abc


class Net(metaclass=abc.ABCMeta):
    """
    Abstract class that is the super class of all models  
    """ 

    @abc.abstractmethod
    def setup_model(self):
        return


class EffnetB1(Net):
    """
    Efficient Net Version B1 implementation
    """
    def __init__(self, train_generator, initial_parameters=None):
        if initial_parameters is None:
            self.initial_parameters = {}
        else:
            self.initial_parameters = initial_parameters
        self.train_generator = train_generator
        self.model = self.setup_model()

    def setup_model(self):
        """
        Build the complete Model.
        
        Returns
        -------
        Net
            The complete Model.
        """
        base_model = self._setup_base_model()
        return self._setup_final_layers(base_model)

    def _setup_base_model(self):
        """
        Build the EfficientNet layers
        
        Returns
        -------
        Net
            Model composed by Efficient Net layers
        """
        base_model = efn.EfficientNetB1(weights='imagenet', include_top=False)
        # fix the feature extraction part of the model
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        return base_model

    def _setup_final_layers(self, base_model):
        """
        Build the last layers
        
        Parameters
        ----------
        base_model : Net
            Model composed by Efficient Net layers
        
        Returns
        -------
        Net
            The complete model
        """
        x = GlobalAveragePooling2D()(base_model.output)
        predictions = Dense(len(self.train_generator.class_indices),
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(lr=0.01),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        return model


class Prototype(Net):
    """
    Basic model used for baseline and test purposes.
    """
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
            Dense(len(self.train_generator.class_indices), activation='softmax')
            ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


class EffnetB7(Net):
    """
    Efficient Net Version B7 implementation
    """

    def __init__(self, train_generator, initial_parameters=None):
        if initial_parameters is None:
            self.initial_parameters = {}
        else:
            self.initial_parameters = initial_parameters
        self.train_generator = train_generator
        self.model = self.setup_model()

    def setup_model(self):
        """
        Build the complete Model.
        
        Returns
        -------
        Net
            The complete Model.
        """
        base_model = self._setup_base_model()
        return self._setup_final_layers(base_model)

    def _setup_base_model(self):
        """
        Build the EfficientNet layers
        
        Returns
        -------
        Net
            Model composed by Efficient Net layers
        """
        base_model = efn.EfficientNetB7(weights='imagenet', include_top=False)
        # fix the feature extraction part of the model
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        return base_model

    def _setup_final_layers(self, base_model):
        """
        Build the last layers

        Parameters
        ----------
        base_model : Net
            Model composed by Efficient Net layers

        Returns
        -------
        Net
            The complete model
        """
        x = GlobalAveragePooling2D()(base_model.output)
        predictions = Dense(len(self.train_generator.class_indices),
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(lr=0.01),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        return model
