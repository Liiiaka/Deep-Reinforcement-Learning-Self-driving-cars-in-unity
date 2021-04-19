from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines.bench import Monitor
from stable_baselines import logger
import stable_baselines.ppo2.ppo2 as ppo2

import gym

import os

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def main():
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(width=1800, height=900, time_scale=12.0)

    # create the environment
    env_directory = "C:\\Users\\wolff\Desktop\\MLAgents\\envs_continuous\\DeepReinforcementLearning"
    unity_env = UnityEnvironment(env_directory, side_channels=[config_channel])
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=False)
    ppo = ppo2.PPO2("MlpPolicy", env)
    ppo.learn(
        total_timesteps=10000000,
        log_interval=10,
    )

if __name__ == '__main__':
    main()