from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense


class ContinuousActorCriticModel(Model):
    """Implementation of an actor-critic model with a continuous action space."""

    def __init__(self, num_observations, num_actions, actor_hidden_units=[256, 256, 256], critic_hidden_units=[128, 128],
                 actor_activation_function='relu', critic_activation_function='relu'):
        super(ContinuousActorCriticModel, self).__init__()

        # initialize the spaces
        self._num_obs = num_observations
        self._num_actions = num_actions

        # create the actor and critic
        self.actor = self._create_actor(actor_hidden_units, actor_activation_function)
        self.critic = self._create_critic(critic_hidden_units, critic_activation_function)

    def __call__(self, state_batch):
        mu_batch, sigma_batch = self.actor(state_batch)
        value_estimate_batch = self.critic(state_batch)
        return {'mu': tf.squeeze(mu_batch), 'sigma': tf.squeeze(sigma_batch), 'value_estimate': tf.squeeze(value_estimate_batch)}

    def save(self, save_path):
        saving_path_actor = Path(save_path + '_actor')
        saving_path_actor.mkdir(parents=True, exist_ok=True)
        self.actor.save(saving_path_actor)

        saving_path_critic = Path(save_path + '_critic')
        saving_path_critic.mkdir(parents=True, exist_ok=True)
        self.critic.save(saving_path_critic)

    def _create_actor(self, actor_hidden_units, actor_activation_function):
        state_input = Input(shape=self._num_obs)

        # create the hidden layers
        next_input = state_input
        for i in range(len(actor_hidden_units)):
            units = actor_hidden_units[i]
            activation = actor_activation_function
            name = f'actor_dense_{i}'
            next_input = Dense(units, activation=activation, name=name)(next_input)

        # create the output layers
        mu = Dense(self._num_actions, activation='tanh', name='actor_mu')(next_input)
        sigma = tf.exp(Dense(self._num_actions, activation=None, name='actor_sigma')(next_input))
        return Model(inputs=state_input, outputs=[mu, sigma])

    def _create_critic(self, critic_hidden_units, critic_activation_function):
        state_input = Input(shape=self._num_obs)

        # create the hidden layers
        next_input = state_input
        for i in range(len(critic_hidden_units)):
            units = critic_hidden_units[i]
            activation = critic_activation_function
            name = f'critic_dense_{i}'
            next_input = Dense(units, activation=activation, name=name)(next_input)

        # create the output layers
        value_estimate = Dense(1, name='critic_value_estimate')(next_input)
        return Model(inputs=state_input, outputs=value_estimate)
