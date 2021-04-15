import time

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from models import ContinuousActorCriticModel
from policies import ContinuousPolicy
from trainer import PPOTrainer


def main():
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(width=1800, height=900, time_scale=10.0)

    # create the environment
    env_directory = "C:\\Users\\wolff\Desktop\\MLAgents\\envs_continuous\\DeepReinforcementLearning"
    unity_env = UnityEnvironment(env_directory, side_channels=[config_channel])
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=False)

    # create the model
    model = ContinuousActorCriticModel(env.observation_space.shape[0], env.action_space.shape[0])

    # create the policy
    policy = ContinuousPolicy(model)

    # create the trainer
    trainer = PPOTrainer(policy, env, learning_rate=0.0003, learning_rate_critic=0.0003, episodes=30000, sample_size=512, batch_size=512, gamma=0.995, clipping_value=0.2)
    trainer.set_saving_params(f'\\saved_models\\{int(time.time())}', 50, True)

    trainer.train()


if __name__ == '__main__':
    main()