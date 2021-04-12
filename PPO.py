import numpy as np

import gym

from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam
from ReplayBuffer import Buffer

import numba as nb
from tensorboardX import SummaryWriter

class Agent:
    def __init__(self, env, buffer_size, batch_size, lr, beta, epsilon, lambd):
        self.env = env
        self.replayBuffer = Buffer(buffer_size, batch_size, env.action_space.n, env.observation_space.n)
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.lambd = lambd
        self.actor = self.create_actor()
        self.critic = self.create_critic()

    def _get_action(self)

    def create_actor(self):
        state_input = Input(shape(self.env.observation_space.n,))
        advantage = Input(shape(1,))
        old_prediction = Input(shape=(self.env.action_space.n,))

        x = Dense(256, activation='tanh')(x)
        x = Dense(256, activation='tanh')(x)
        x = Dense(256, activation='tanh')(x)

        out_actions = Dense(self.env.action_space.n, activation='softmax', name='output')(x)

        model = Model(input=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=self.lr), loss=proximal_policy_optimization_loss(advantage=advantage, old_prediction=old_prediction))

        print("Actor Model: ")
        model.summary()

        return model

    def create_critic(self):

        state_input = Input(shape=(self.env.observation_space.n,))
        x = Dense(128, activation='tanh')(state_input)
        x = Dense(128, activation='tanh')(x)
        output_layer = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[output_layer])
        model.compile(optimizer=Adam(lr=self.lr), loss='mse')

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