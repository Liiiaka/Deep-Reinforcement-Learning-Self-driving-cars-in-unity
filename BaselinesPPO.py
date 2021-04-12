from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import logger
import stable_baselines.ppo2.ppo2 as ppo2

import os

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def make_unity_env(env_directory, num_env, visual, start_index=0):
    """
    Create a wrapped, monitored Unity environment.
    """
    def make_env(rank, use_visual=True): # pylint: disable=C0111
        def _thunk():
            unity_env = UnityEnvironment(env_directory, no_graphics=True)
            env = UnityToGymWrapper(unity_env, rank, uint8_visual=use_visual)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    if visual:
        return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else:
        rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        return DummyVecEnv([make_env(rank, use_visual=False)])

def main():
    env_directory = "C:\\Users\\wolff\\Desktop\\MLAgents\\envs\\DeepReinforcementLearning"
    unity_env = UnityEnvironment(env_directory)
    env = UnityToGymWrapper(unity_env)
    print(env.action_space)
    ppo = ppo2.PPO2("MlpPolicy", env)
    ppo.learn(
        total_timesteps=100000,
    )

if __name__ == '__main__':
    main()