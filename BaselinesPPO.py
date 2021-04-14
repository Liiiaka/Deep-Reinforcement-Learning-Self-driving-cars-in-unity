from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

import os

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def main():
    env_directory = "./unity_envs/PyTrainEnv_RandomParkingLots"
    unity_env = UnityEnvironment(env_directory)
    env = UnityToGymWrapper(unity_env)
    print(env.action_space)
    print(env.action_space.shape)
    print(env.observation_space.shape)

if __name__ == '__main__':
    main()