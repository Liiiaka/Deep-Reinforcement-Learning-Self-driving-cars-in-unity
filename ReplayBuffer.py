import numpy as np
import Tensorflow as Tf

class Buffer:
    def __init__(self, capacity, batch_size, num_states, num_actions):
        self.capacity = capacity
        self.batch_size = batch_size

        self.state_buffer = np.array([])
        self.action_buffer = np.array([])
        self.next_state_buffer = np.array([])
        self.reward_buffer = np.array([])
        self.dones_buffer = np.array([])

    def add(self, state, action, reward, next_state, done):
        if len(self.state_buffer) > 0:
            self.state_buffer = np.concatenate((self.state_buffer, state), axis=0)
            self.action_buffer = np.concatenate((self.action_buffer, action.numpy()), axis=0)
            self.next_state_buffer = np.concatenate((self.next_state_buffer, next_state), axis=0)
            self.reward_buffer = np.concatenate((self.reward_buffer, [reward]), axis=0)
            self.dones_buffer = np.concatenate((self.dones_buffer, [done]), axis=0)
        else:
            self.state_buffer = np.array(state)
            self.action_buffer = np.array(action)
            self.next_state_buffer = np.array(next_state)
            self.reward_buffer = np.array([reward])
            self.dones_buffer = np.array([done])

        
        if len(self.state_buffer) > self.capacity:
            self.state_buffer = np.delete(self.state_buffer, 0, 0)
            self.action_buffer = np.delete(self.action_buffer, 0, 0)
            self.next_state_buffer = np.delete(self.next_state_buffer, 0, 0)
            self.reward_buffer = np.delete(self.reward_buffer, 0, 0)
            self.dones_buffer = np.delete(self.dones_buffer, 0, 0)

    def get_batch(self):
        batch_indices = np.random.choice(len(self.state_buffer), self.batch_size)

        states = tf.convert_to_tensor(self.state_buffer[batch_indices])
        actions = tf.convert_to_tensor(self.action_buffer[batch_indices])
        rewards = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        rewards = tf.cast(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        dones = self.dones_buffer[batch_indices]

        return states, actions, rewards, next_states, dones