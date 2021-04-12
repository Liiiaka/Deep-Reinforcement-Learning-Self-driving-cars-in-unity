import numpy as np
import tensorflow as tf

class Buffer:

    def __init__(self, capacity, batch_size, num_states, num_actions):
        self.capacity = capacity
        self.batch_size = batch_size
        self.current_index = 0

        self.state_buffer = np.zeros(shape=(self.capacity, num_states))
        self.action_buffer = np.zeros(shape=(self.capacity, num_actions))
        self.next_state_buffer = np.zeros(shape=(self.capacity, num_states))
        self.reward_buffer = np.zeros(shape=(self.capacity,))
        self.dones_buffer = np.zeros(shape=(self.capacity,))

    def add(self, state, action, reward, next_state, done):
        """Adds a new sample to the buffer."""
        index = self.current_index % self.capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.next_state_buffer[index] = next_state
        self.reward_buffer[index] = reward
        self.dones_buffer[index] = done

        self.current_index = self.current_index + 1

    def reset(self):
        """Resets the buffer."""
        self.state_buffer = np.zeros_like(self.state_buffer)
        self.action_buffer = np.zeros_like(self.action_buffer)
        self.next_state_buffer = np.zeros_like(self.next_state_buffer)
        self.reward_buffer = np.zeros_like(self.reward_buffer)
        self.dones_buffer = np.zeros_like(self.dones_buffer)

        self.current_index = 0

    def get_dataset(self):
        """Creates the dataset with all available data and returns it."""
        index = min(self.capacity, self.current_index)
        dataset = tf.data.Dataset.from_tensor_slices((self.state_buffer[:index],
                                                      self.action_buffer[:index],
                                                      self.next_state_buffer[:index],
                                                      self.reward_buffer[:index],
                                                      self.dones_buffer[:index]))
        return dataset
