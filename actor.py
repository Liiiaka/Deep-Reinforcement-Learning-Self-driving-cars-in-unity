import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from gym.spaces import MultiDiscrete


class Actor():
    """Implements the base actor."""

    def __init__(self, observation_space, action_space, epsilon, hidden_units=[256, 256, 256], activation_function='tanh', clipping_value=0.2):
        if action_space is MultiDiscrete:
            self._model = MultiDiscreteActor(observation_space, action_space, hidden_units, activation_function)
        else:
            raise ValueError("Currently just MultiDiscrete action spaces are supported.")
        self._epsilon = epsilon
        self.clipping_value = clipping_value

    def __call__(self, inputs):
        return self._model(inputs)

    def policy(self, state, is_test=False):
        """Returns the actions and the related probabilities."""
        logits = self._model(state)
        if type(logits) is list:
            actions, probs = []
            for tmp_logits in logits:
                tmp_actions, tmp_probs = self._get_action(tmp_logits, is_test)
                actions.append(tmp_actions)
                probs.append(tmp_logits)
        else:
            actions, probs = self._get_action(logits, is_test)
        return actions, probs

    def _get_action(self, logits, is_test):
        if tf.is_tensor(logits):
            logits = logits.numpy()

        local_epsilon = 0 if is_test else self.epsilon
        greedy_prob = 1 - local_epsilon
        ungreedy_prob_per_action = local_epsilon / logits.shape[-1]
        if np.random.random() > local_epsilon:
            # greedy selection
            action = logits.argmax(axis=-1)
            prob = greedy_prob + ungreedy_prob_per_action
        else:
            action = np.random.randint(logits.shape[-1], logits.shape[0])
            prob = local_epsilon / logits.shape[-1]
        return action, prob

    def _get_probs(self, state, action):
        logits = self._model(state)
        if type(logits) is list:
            probs = []
        else:
            probs = self._get_probs_for_single_action(logits, action)
        greedy_action = logits.argmax(axis=-1)
        pass

    def _get_probs_for_single_action(self, logits, action):
        pass

    def train(self, state_batch, action_batch, old_prob_batch, advantage_batch, optimizer):
        with tf.GradientTape() as tape:
            # calculate the probability ratio
            new_prob = self._get_probs(state_batch, action_batch)
            prob_ratio = tf.exp(new_prob - tf.cast(old_prob_batch, dtype=tf.float32))

            # calculate the loss - PPO
            unclipped_loss = prob_ratio * tf.expand_dims(advantage_batch, 1)
            clipped_loss = tf.clip_by_value(prob_ratio, 1 - self.clipping_value, 1 + self.clipping_value) * tf.expand_dims(
                advantage_batch, 1)
            loss = -tf.reduce_mean(tf.minimum(unclipped_loss, clipped_loss))
            gradients = tape.gradient(loss, self.model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss


class MultiDiscreteActor(Model):
    """Implements the neural network model for a multi categorical actor."""

    def __init__(self, observation_space, action_space, hidden_units, activation_function):
        self._input_size = observation_space.shape[0]
        self._action_branch_sizes = action_space.nvec
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
        outputs = [Dense(units, activation='softmax') for units in self._action_branch_sizes]

        return Model(inputs=[input], outputs=outputs)

    def __call__(self, inputs):
        return self.model(inputs)
