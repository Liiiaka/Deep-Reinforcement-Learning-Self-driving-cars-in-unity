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
            actions, log_probs = self._get_action(state)
            next_state, reward, done, info = self.env.step(actions)
            self.buffer.add(state, actions, reward, next_state, log_probs, done)
            if done:
                next_state = self.env.reset()
            state = next_state
            

    def _get_action(self, state, is_test=False):
        # get the network output
        action1, action2, action3 = self.actor(state)
        if tf.is_tensor(action1):
            action1 = action1.numpy()
            action2 = action2.numpy()
            action3 = action3.numpy()

        local_epsilon = 0 if is_test else self.epsilon
        if random.random() > local_epsilon:
            # greedy selection of actions
            actual_action1 = np.argmax(action1, axis=-1)
            log_probs1 = np.asarray([np.log(local_epsilon + CONST_TO_PREVENT_DIVISION_BY_ZERO)] * self.env.action_space.nvec[0],
                                  dtype=np.float32)

            actual_action2 = np.argmax(action2, axis=-1)
            log_probs2 = np.asarray([np.log(local_epsilon + CONST_TO_PREVENT_DIVISION_BY_ZERO)] * self.env.action_space.nvec[1],
                                  dtype=np.float32)

            actual_action3 = np.argmax(action3, axis=-1)
            log_probs3 = np.asarray([np.log(local_epsilon + CONST_TO_PREVENT_DIVISION_BY_ZERO)] * self.env.action_space.nvec[2],
                                  dtype=np.float32)
            
        else:
            # selection any action
            actual_action1 = np.random.randint(self.env.action_space.nvec[0])
            actual_action2 = np.random.randint(self.env.action_space.nvec[1])
            actual_action3 = np.random.randint(self.env.action_space.nvec[2])

            log_probs1 = np.asarray([np.log(1 - local_epsilon)] * self.env.action_space.nvec[0], dtype=np.float32)
            log_probs2 = np.asarray([np.log(1 - local_epsilon)] * self.env.action_space.nvec[1], dtype=np.float32)
            log_probs3 = np.asarray([np.log(1 - local_epsilon)] * self.env.action_space.nvec[2], dtype=np.float32)

        return (actual_action1, actual_action2, actual_action3), (log_probs1, log_probs2, log_probs3)

    def _create_actor(self):
        state_input = Input(shape=(self.observation_space,))
        x = Dense(256, activation='tanh')(state_input)
        x = Dense(256, activation='tanh')(x)
        x = Dense(256, activation='tanh')(x)
        out_actions = Dense(self.env.action_space.nvec[0], activation='softmax', name='output')(x)
        out_actions1 = Dense(self.env.action_space.nvec[1], activation='softmax', name='output2')(x)
        out_actions2 = Dense(self.env.action_space.nvec[2], activation='softmax', name='output3')(x)

        model = Model(input=[state_input], outputs=[out_actions, out_actions1, out_actions2])
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

    @tf.function
    def train_policy_network(agent, state, actions, log_probs, advantage, optimizer):
        """Trains the policy network with PPO clipped loss."""

        with tf.GradientTape() as tape:
            # calculate the probability ratio
            new_log_prob = self.flowing_log_prob(state, action)
            prob_ratio = tf.exp(new_log_prob - tf.cast(log_prob, dtype=tf.float32))

            # calculate the loss - PPO
            unclipped_loss = prob_ratio * tf.expand_dims(advantage, 1)
            clipped_loss = tf.clip_by_value(prob_ratio, 1 - CLIPPING_VALUE, 1 + CLIPPING_VALUE) * tf.expand_dims(advantage, 1)
            loss = -tf.reduce_mean(tf.minimum(unclipped_loss, clipped_loss))
            gradients = tape.gradient(loss, agent.model.actor.trainable_variables)

        optimizer.apply_gradients(zip(gradients, agent.model.actor.trainable_variables))
        return loss

    def flowing_log_prob(state, action):
        logits = self.actor(state)

    @staticmethod
    def proximal_policy_optimization_loss(advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = K.sum(y_true * y_pred, axis=-1)
            old_prob = K.sum(y_true * old_prediction, axis=-1)
            r = prob/(old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
        return loss