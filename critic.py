from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


class Critic:
    """Implements the base critic."""

    def __init__(self, observation_space, hidden_units=[128, 128], activation_function='tanh'):
        self._input_size = observation_space.shape[0]
        self._hidden_units = hidden_units
        self._activation_function = activation_function
        self.model = self._create_model()

    def _create_model(self):
        # define input layer
        input = Input(shape=(self._input_size,))
        # define hidden layers
        x = input
        for units in self._hidden_units:
            x = Dense(units, activation=self._activation_function)(x)
        # define output layers
        output = Dense(1, activation='softmax')

        return Model(inputs=[input], outputs=[output])

    def __call__(self, inputs):
        return self.model(inputs)
