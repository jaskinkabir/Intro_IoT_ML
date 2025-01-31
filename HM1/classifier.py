import keras

from keras.api import layers
from keras.api import ops

class Classifier:
    def __init__(self, input_shape, hidden_layers, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        self.model = keras.Sequential(
            [
                layers.InputLayer(shape=input_shape),
                layers.Flatten(),
                *[
                    layers.Dense(num_units, activation='relu', name=f'hlayer_{i}') for i, num_units in enumerate(hidden_layers)
                ],
                layers.Dense(num_classes, activation='softmax', name='output_layer')
            ]
        )
    def train(self, x, y, validation_data, epochs=10):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.hist = self.model.fit(x, y, epochs=epochs, validation_data=validation_data)
        
    def predict(self, x):
        return self.model.predict(x)