import csv
import logging

from algorithms.atari.PPO import PPOAgent
from algorithms.deepmindcs.D4PG import D4PGAgent
from algorithms.deepmindcs.mujoco_env import get_env
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from mingpt.utils import get_action_infor
from collections import deque
import random
import torch
import pickle
import argparse
from create_dataset import create_dataset
from algorithms.atari.DQN import RainbowAgent

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Pong')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--action_space',type=int,default=6)
parser.add_argument('--action_type',type=str,default='atari')
#
parser.add_argument('--target_model_type',type=str, default='PPO')
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./attacks/datasets/FGSM_pong_data/')
parser.add_argument('--policy_path',type=str, default='./algorithms/models/ppo_Pong_final.pth')
args = parser.parse_args()

set_seed(args.seed)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps, deltas_data):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.deltas_data = deltas_data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        deltas = torch.tensor(np.array(self.deltas_data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        deltas = deltas / 255.

        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps, deltas


class ConStateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps, deltas_data, action_dim=11, action_low=None, action_high=None):
        self.block_size = block_size
        self.action_dim = action_dim
        self.vocab_size = action_dim
        self.data = data
        self.actions = actions  
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.deltas_data = deltas_data
        self.action_low = action_low  
        self.action_high = action_high

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size

        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1)
        states = states 
        deltas = torch.tensor(np.array(self.deltas_data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1)
        deltas = deltas 

        # Continuous actions: (block_size, 11)
        actions = torch.tensor(np.array(self.actions[idx:done_idx]), dtype=torch.float32)
        # Optional: normalize to [-1, 1]
        if self.action_low is not None and self.action_high is not None:
            low = torch.tensor(self.action_low, dtype=torch.float32).unsqueeze(0).expand_as(actions)
            high = torch.tensor(self.action_high, dtype=torch.float32).unsqueeze(0).expand_as(actions)
            actions = 2.0 * (actions - low) / (high - low) - 1.0
            actions = actions.clamp(-1.0, 1.0)

        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps, deltas

obss, actions, returns, done_idxs, rtgs, timesteps, deltas = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer,action_type=args.action_type)

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

if args.action_type=='atari' or args.action_type=='gfootball':
    train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps,deltas)
elif args.action_type=='mujoco':
    env = get_env(args.game)
    action_spec = env.action_spec
    train_dataset = ConStateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps,deltas,action_dim=action_spec.shape[0], action_low=-1, action_high=1)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
model = GPT(mconf,args.action_type,args.target_model_type)

target_model=None
if args.action_type=='atari' or args.action_type=='gfootball':
    action_dim=get_action_infor(args.game)
    target_model = PPOAgent(action_dim)
    target_model.load(args.policy_path)
elif args.action_type=='mujoco':
    env=get_env(args.game)
    action_spec = env.action_spec
    target_model = D4PGAgent(action_spec.shape[0], action_spec)



# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps),target_model_taype=args.target_model_type)
trainer = Trainer(model, train_dataset, None, tconf, target_model, action_type=args.action_type)

trainer.train()
