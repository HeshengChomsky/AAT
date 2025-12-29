import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import logging
from mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
from algorithms.deepmindcs.mujoco_env import PixelWrapper
from dm_control import suite
from dm_control.suite.wrappers import pixels
import gym
import argparse
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Env
from mingpt.trainer_atari import Args
from algorithms.atari.PPO import PPOAgent
from create_dataset import load_dataset_infor
from algorithms.atari.Atarienv import AtariWrapper
from algorithms.atari.DQN import RainbowAgent
from mingpt.utils import get_action_infor
from algorithms.deepmindcs.mujoco_env import get_env
from algorithms.deepmindcs.D4PG import D4PGAgent
# from algorithms.gfootball.Gfootball_env import make_env_from_args
# import gfootball.env as football_env
import torch
import numpy as np


def evaluate_trained_model(model, target_model, game='Pong', seed=123, episodes=10, target_return=20, max_timestep=90, model_type=None, is_attack=True, device=None,action_type='atari'):
    device = torch.device('cuda' if torch.cuda.is_available() else ('cpu' if device is None else device))
    if action_type=='mujoco':
        if hasattr(model, 'module'):
            model = model.module
        model = model.to(device)
        model.train(False)
        env_raw = suite.load(game, 'stand')
        env_pixels = pixels.Wrapper(env_raw, pixels_only=True)
        env = PixelWrapper(env_pixels)
        T_rewards = []
        done=True
        if is_attack:
            for _ in range(episodes):
                state = env.reset()
                state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
                rtgs = [target_return]
                delta = sample(model, state, 1, temperature=1.0, sample=True, actions=None,
                               rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1),
                               timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device), target_model=None)
                state_cop = state.squeeze()
                delta_cop = torch.reshape(delta, (state_cop.shape[0], state_cop.shape[1], state_cop.shape[2]))
                j = 0
                all_states = state
                actions = []
                deltas = []
                while True:
                    if done:
                        state, reward_sum, done= env.reset(), 0, False
                    adv_state = np.clip((state_cop + delta_cop).cpu().numpy() * 255.,0,255)
                    adv_state = torch.from_numpy(adv_state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = target_model.actor(adv_state)
                        output = output.clamp(-1, 1)
                        action= output.squeeze(0).cpu().numpy()
                    actions += [action]
                    deltas += [delta_cop.cpu().numpy()]
                    state, reward, done, _ = env.step(action)
                    reward_sum += reward
                    j += 1
                    if done:
                        T_rewards.append(reward_sum)
                        break
                    state = state.unsqueeze(0).unsqueeze(0).to(device)
                    all_states = torch.cat([all_states, state], dim=0)
                    rtgs += [rtgs[-1] - (-1 * reward)]
                    delta = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                                   actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0),
                                   rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1),
                                   timesteps=(min(j, max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)),
                                   delta=torch.tensor(np.array(deltas), dtype=torch.float32).to(device).unsqueeze(0), target_model=None)
                    delta_cop = torch.reshape(delta, (state_cop.shape[0], state_cop.shape[1], state_cop.shape[2]))
        else:
            for i in range(episodes):
                while True:
                    if done:
                        state, reward_sum, done= env.reset(), 0, False
                        # state = state.type(torch.float32).to(device)
                    with torch.no_grad():
                        state=state.unsqueeze(dim=0).to(device)
                        output = target_model.actor(state)
                        output = output.clamp(-1, 1)
                        action=output.squeeze(0).cpu().numpy()
                    state, reward, done, _ = env.step(action)
                    reward_sum += reward
                    if done:
                        T_rewards.append(reward_sum)
                        break
                    state = state

        avg = sum(T_rewards) / float(episodes)
        print(f"Evaluation over {episodes} episodes: Avg Reward {avg:.2f}")
        model.train(True)
        return avg
    elif action_type=='gfootball':
        _env = football_env.create_environment(env_name=game, representation='pixels', stacked=True, with_checkpoints=False, render=False, channels_order='last')
        T_rewards = []
        done = True
        if is_attack:
            for _ in range(episodes):
                obs = _env.reset()
                obs_t = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).div(255).to(device)
                state = obs_t.unsqueeze(0).unsqueeze(0)
                rtgs = [target_return]
                delta = sample(model.module if hasattr(model, 'module') else model, state, 1, temperature=1.0, sample=True, actions=None,
                               rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1),
                               timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device), target_model=None)
                state_cop = state.squeeze()
                delta_cop = torch.reshape(delta, (state_cop.shape[0], state_cop.shape[1], state_cop.shape[2]))
                j = 0
                all_states = state
                actions = []
                deltas = []
                while True:
                    if done:
                        obs, reward_sum, done = _env.reset(), 0, False
                        obs_t = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).div(255).to(device)
                        state_cop = obs_t
                    adv_state = np.clip((state_cop + delta_cop).cpu().numpy() * 255., 0, 255)
                    if model_type == 'PPO' or model_type == 'TRPO':
                        adv_state_t = torch.from_numpy(adv_state).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits, _ = target_model.model(adv_state_t)
                            action = torch.argmax(logits, dim=-1).item()
                    else:
                        action = target_model.select_action(adv_state)
                    actions += [action]
                    deltas += [delta_cop.cpu().numpy()]
                    obs, reward, done, _ = _env.step(action)
                    reward_sum += reward
                    j += 1
                    if done:
                        T_rewards.append(reward_sum)
                        break
                    obs_t = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).div(255).to(device)
                    state = obs_t.unsqueeze(0).unsqueeze(0)
                    all_states = torch.cat([all_states, state], dim=0)
                    rtgs += [rtgs[-1] - (-1 * reward)]
                    delta = sample(model.module if hasattr(model, 'module') else model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                                   actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0),
                                   rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1),
                                   timesteps=(min(j, max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)),
                                   delta=torch.tensor(np.array(deltas), dtype=torch.float32).to(device).unsqueeze(0), target_model=None)
                    delta_cop = torch.reshape(delta, (state_cop.shape[0], state_cop.shape[1], state_cop.shape[2]))
        else:
            for i in range(episodes):
                while True:
                    if done:
                        obs, reward_sum, done = _env.reset(), 0, False
                    if model_type == 'PPO' or model_type == 'TRPO':
                        obs_t = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).div(255).to(device)
                        obs_t = obs_t.unsqueeze(0)
                        with torch.no_grad():
                            logits, _ = target_model.model(obs_t)
                            action = torch.argmax(logits, dim=-1).item()
                    else:
                        action = target_model.select_action(obs)
                    obs, reward, done, _ = _env.step(action)
                    reward_sum += reward
                    if done:
                        T_rewards.append(reward_sum)
                        break
        _env.close()
        avg = sum(T_rewards) / float(episodes)
        print(f"Evaluation over {episodes} episodes: Avg Reward {avg:.2f}")
        model.train(True)
        return avg
    else:
        if hasattr(model, 'module'):
            model = model.module
        model = model.to(device)
        model.train(False)
        args = Args(game.lower(), seed)
        T_rewards = []
        done = True
        if is_attack:
            env = Env(args)
            env.eval()
            for _ in range(episodes):
                state = env.reset()
                state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
                rtgs = [target_return]
                delta = sample(model, state, 1, temperature=1.0, sample=True, actions=None,
                               rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1),
                               timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device), target_model=None)
                state_cop = state.squeeze()
                delta_cop = torch.reshape(delta, (state_cop.shape[0], state_cop.shape[1], state_cop.shape[2]))
                j = 0
                all_states = state
                actions = []
                deltas = []
                while True:
                    if done:
                        state, reward_sum, done = env.reset(), 0, False
                    adv_state = np.clip((state_cop + delta_cop).cpu().numpy() * 255.,0,255)
                    if model_type=='PPO' or model_type=='TRPO':
                        adv_state = torch.from_numpy(adv_state).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits, _ = target_model.model(adv_state)
                            action = torch.argmax(logits, dim=-1).item()
                    else:
                        action = target_model.select_action(adv_state)
                    actions += [action]
                    deltas += [delta_cop.cpu().numpy()]
                    state, reward, done = env.step(action)
                    reward_sum += reward
                    j += 1
                    if done:
                        T_rewards.append(reward_sum)
                        break
                    state = state.unsqueeze(0).unsqueeze(0).to(device)
                    all_states = torch.cat([all_states, state], dim=0)
                    rtgs += [rtgs[-1] - (-1 * reward)]
                    delta = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                                   actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0),
                                   rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1),
                                   timesteps=(min(j, max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)),
                                   delta=torch.tensor(np.array(deltas), dtype=torch.float32).to(device).unsqueeze(0), target_model=None)
                    delta_cop = torch.reshape(delta, (state_cop.shape[0], state_cop.shape[1], state_cop.shape[2]))
        else:
            name = game + 'NoFrameskip-v4'
            env = AtariWrapper(gym.make(name))
            state = None
            reward_sum = 0
            # state = state.type(torch.float32).to(device)
            for i in range(episodes):
                while True:
                    if done:
                        state, reward_sum, done= env.reset(), 0, False
                        # state = state.type(torch.float32).to(device)
                    if model_type == 'PPO' or model_type == 'TRPO':
                        # action, _, _ = target_model.select_action(state)
                        state=torch.from_numpy(state).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits, _ = target_model.model(state)
                            action = torch.argmax(logits, dim=-1).item()
                    else:
                        action = target_model.select_action(state)
                    state, reward, done,_ = env.step(action)
                    reward_sum += reward
                    if done:
                        print(reward_sum)
                        T_rewards.append(reward_sum)
                        break
                    state = state

        env.close()
        avg = sum(T_rewards) / float(episodes)
        print(f"Evaluation over {episodes} episodes: Avg Reward {avg:.2f}")
        model.train(True)
        return avg


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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--context_length', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--num_buffers', type=int, default=50)
    parser.add_argument('--game', type=str, default='Pong')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--action_space', type=int, default=6)
    parser.add_argument('--action_type', type=str, default='atari')
    # parser.add_argument('aat_model_path', type=str,default='models/aat_model_final.pth')
    #
    parser.add_argument('--target_model_type', type=str, default='PPO')
    parser.add_argument('--trajectories_per_buffer', type=int, default=10,
                        help='Number of trajectories to sample from each of the buffers.')
    parser.add_argument('--data_dir_prefix', type=str, default='./attacks/datasets/FGSM_Pong_data/')
    parser.add_argument('--policy_path', type=str, default='./algorithms/models/ppo_Pong_final.pth')
    args = parser.parse_args()

    obss, actions, returns, done_idxs, rtgs, timesteps, deltas = load_dataset_infor(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer,action_type=args.action_type)

    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.action_type == 'atari' or args.action_type == 'gfootball':
        train_dataset = StateActionReturnDataset(obss, args.context_length * 3, actions, done_idxs, rtgs, timesteps,
                                                 deltas)
    elif args.action_type == 'mujoco':
        env = get_env(args.game)
        action_spec = env.action_spec
        train_dataset = ConStateActionReturnDataset(obss, args.context_length * 3, actions, done_idxs, rtgs, timesteps,
                                                    deltas, action_dim=action_spec.shape[0], action_low=-1,
                                                    action_high=1)



    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                      n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
    model = GPT(mconf, args.action_type, args.target_model_type)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model).to(device)

    state = torch.load('models/aat_model_final.pth')
    model.module.load_state_dict(state)
    model.module.eval()

    target_model = None
    if args.action_type == 'atari' or args.action_type=='gfootball':
        action_dim = get_action_infor(args.game)
        target_model = PPOAgent(action_dim)
        target_model.load(args.policy_path)
    elif args.action_type == 'mujoco':
        env = get_env(args.game)
        action_spec = env.action_spec
        target_model = D4PGAgent(action_spec.shape[0], action_spec)

    eval_reward = evaluate_trained_model(model, target_model, args.game, args.seed, episodes=args.epochs, target_return=20, max_timestep=args.context_length, model_type=args.target_model_type,is_attack=True,action_type=args.action_type)

if __name__ == '__main__':
    main()

