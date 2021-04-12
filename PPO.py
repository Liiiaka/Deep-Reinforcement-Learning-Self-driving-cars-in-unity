import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from keras import backend as K
from tensorflow.keras.optimizers import Adam
from buffer import Buffer

import numba as nb
from tensorboardX import SummaryWriter

CONST_TO_PREVENT_DIVISION_BY_ZERO = 1e-7

class Agent:

    def __init__(self, env, buffer_size, batch_size, lr, beta, epsilon, lambd):
        self.env = env
        self._action_space = self.env.action_space.shape[0] # TODO: We have 3 actions with 7, 7, and 2 possibilities, how do we get these?
        self._observation_space = self.env.observation_space.shape[0]
        self.buffer = Buffer(buffer_size, batch_size, self._action_space, self._observation_space)
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.lambd = lambd
        self._batch_size = batch_size
        self.actor = self._create_actor()
        self.critic = self._create_critic()

    def train(self, num_epochs):
        for e in range(1, num_epochs+1):
            # sample trajectories
            samples = self._get_trajectories()
            #
            pass

    def _get_trajectories(self):
        self.buffer.reset()
        state = np.reshape(self.env.reset(), (1, self._observation_space))
        for _ in range(self.buffer_size):
            action = self._get_action(state)
            pass

    def _get_action(self, state, is_test=False):
        # get the network output
        logits = self.actor(state)
        if tf.is_tensor(logits):
            logits = logits.numpy()

        local_epsilon = 0 if is_test else self.epsilon
        if random.random() > local_epsilon:
            # greedy selection of actions
            actions = np.argmax(logits, axis=-1)
            log_probs = np.asarray([np.log(local_epsilon + CONST_TO_PREVENT_DIVISION_BY_ZERO)] * logits.shape[0],
                                  dtype=np.float32)
            pass
        else:
            # selection any action
            actions = [random.ran]

            pass

    def _create_actor(self):
        state_input = Input(shape=(self.observation_space,))
        x = Dense(256, activation='tanh')(state_input)
        x = Dense(256, activation='tanh')(x)
        x = Dense(256, activation='tanh')(x)
        out_actions = Dense(self.env.action_space.n, activation='softmax', name='output')(x)

        model = Model(input=[state_input], outputs=[out_actions])
        print("Actor Model: ")
        model.summary()

        return model

    def _create_critic(self):
        state_input = Input(shape=(self.observation_space,))
        x = Dense(128, activation='tanh')(state_input)
        x = Dense(128, activation='tanh')(x)
        output_layer = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[output_layer])
        print("Critic Model: ")
        model.summary()

        return model

    @staticmethod
    def proximal_policy_optimization_loss(advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = K.sum(y_true * y_pred, axis=-1)
            old_prob = K.sum(y_true * old_prediction, axis=-1)
            r = prob/(old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
        return loss