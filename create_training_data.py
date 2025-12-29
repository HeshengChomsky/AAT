import numpy as np
import torch
import gym
from algorithms.Atarienv import AtariWrapper
from algorithms.deepmindcs.mujoco_env import PixelWrapper
from dm_control import suite
from dm_control.suite.wrappers import pixels
from algorithms.PPO import PPOAgent
from algorithms.deepmindcs.D4PG import D4PGAgent
from attacks.FGSM.FGSM import collected_data
import os
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def created_data(action_type):
    if action_type == 'atari':
        name = 'Pong' + 'NoFrameskip-v4'
        env = AtariWrapper(gym.make(name))
        agent = PPOAgent(env.action_space.n)
        ppo_agent = PPOAgent(env.action_space.n)
        ppo_agent.load('algorithms/models/ppo_pong_final.pth')
        ppo_net=ppo_agent.model.cpu()
        ppo_net.eval()
        path='attacks/datasets/FGSM_Pong_data/'
        os.makedirs(path, exist_ok=True)
        collected_data(ppo_net,env,max_eposides=2,path=path, game='pong',action_type=action_type)
    elif action_type == 'mujoco':
        ENV_NAME = 'hopper'
        env_raw = suite.load(ENV_NAME, 'stand')
        env_pixels = pixels.Wrapper(env_raw, pixels_only=True)
        env = PixelWrapper(env_pixels)
        action_spec = env.action_spec
        agent = D4PGAgent(action_spec.shape[0], action_spec)
        actor_net=agent.actor.cpu()
        actor_net.eval()
        path = 'attacks/datasets/FGSM_hopper_data/'
        os.makedirs(path, exist_ok=True)
        collected_data(actor_net,env,max_eposides=2,path=path,game='hopper',action_type=action_type,model_type='D4PG',action_spec=action_spec)

if __name__ == '__main__':
    created_data('atari')
