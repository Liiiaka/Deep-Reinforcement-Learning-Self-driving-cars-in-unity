import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from keras import backend as K
from tensorflow.keras.optimizers import Adam
from buffer import Buffer
from actor import Actor
from critic import Critic

import numba as nb
from tensorboardX import SummaryWriter

CONST_TO_PREVENT_DIVISION_BY_ZERO = 1e-7


class Agent:

    def __init__(self, env, buffer_size, batch_size, lr, beta, epsilon, lambd):
        self.env = env
        self._action_space = self.env.action_space.shape[0]
        self._observation_space = self.env.observation_space.shape[0]
        self.buffer = Buffer(buffer_size, batch_size, self._action_space, self._observation_space)
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.lambd = lambd
        self._batch_size = batch_size
        self.actor = Actor(self.env.observation_space, self.env.action_space)
        self.critic = Critic(self.env.observation_space)

    def train(self, num_epochs):
        for e in range(1, num_epochs+1):
            # sample trajectories
            self._collect_samples()
            samples = self.buffer.get_dataset()
            samples = samples.batch(self._batch_size)

            for state, actions, next_state, reward, done, log_probs in samples:
                reward_prediction = self.critic(state)
                advantage = reward - reward_prediction

            #
            pass

    def _collect_samples(self):
        self.buffer.reset()
        state = np.reshape(self.env.reset(), (1, self._observation_space))
        for _ in range(self.buffer_size):
            actions, probs = self._get_action(state)
            next_state, reward, done, info = self.env.step(actions)
            self.buffer.add(state, actions, reward, next_state, probs, done)
            if done:
                next_state = self.env.reset()
            state = next_state

    def _get_action(self, state, is_test=False):
        return self.actor.policy(state, is_test)

    @tf.function
    def train_policy_network(agent, state, actions, log_probs, advantage, optimizer):
        """Trains the policy network with PPO clipped loss."""

        with tf.GradientTape() as tape:
            # calculate the probability ratio
            new_log_prob = self.get_log_prob(state, action)
            prob_ratio = tf.exp(new_log_prob - tf.cast(log_prob, dtype=tf.float32))

            # calculate the loss - PPO
            unclipped_loss = prob_ratio * tf.expand_dims(advantage, 1)
            clipped_loss = tf.clip_by_value(prob_ratio, 1 - CLIPPING_VALUE, 1 + CLIPPING_VALUE) * tf.expand_dims(advantage, 1)
            loss = -tf.reduce_mean(tf.minimum(unclipped_loss, clipped_loss))
            gradients = tape.gradient(loss, agent.model.actor.trainable_variables)

        optimizer.apply_gradients(zip(gradients, agent.model.actor.trainable_variables))
        return loss

    def get_log_prob(self, state, action):
        logits = self.actor(state)

    @staticmethod
    def proximal_policy_optimization_loss(advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = K.sum(y_true * y_pred, axis=-1)
            old_prob = K.sum(y_true * old_prediction, axis=-1)
            r = prob/(old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
        return loss