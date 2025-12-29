import os
import numpy as np
import gym
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.distributions import Categorical
from algorithms.atari.Atarienv import AtariWrapper
import torch.nn.functional as F
import argparse
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ENV_NAME='Pong'
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1
LR = 2.5e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 128
EPOCHS = 4
FRAME_STACK = 4
MAX_EPISODES = 13000

class CNNActorCritic(nn.Module):
    def __init__(self, action_dim, in_channels=FRAME_STACK):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        with torch.no_grad():
            x = torch.zeros(1, in_channels, 84, 84)
            n = self.conv(x).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(n, 512),
            nn.ReLU()
        )
        self.policy = nn.Linear(512, action_dim)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, logprob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_advantages(self, gamma, lam):
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values + [0.0], dtype=np.float32)
        advs = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1.0 - dones[t]) * gae
            advs[t] = gae
        returns = advs + values[:-1]
        return returns, advs

    def as_tensors(self):
        states = torch.from_numpy(np.array(self.states)).to(DEVICE)
        actions = torch.from_numpy(np.array(self.actions)).long().to(DEVICE)
        logprobs = torch.from_numpy(np.array(self.logprobs)).float().to(DEVICE)
        return states, actions, logprobs

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

class PPOAgent:
    def __init__(self, action_dim):
        self.model = CNNActorCritic(action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def select_action(self, state):
        s = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, value = self.model(s)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())


    def evaluate_actions(self, states, actions):
        logits, values = self.model(states)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return logprobs, values, entropy

    def update(self, buffer):
        returns_np, advs_np = buffer.compute_returns_advantages(GAMMA, GAE_LAMBDA)
        advs = torch.from_numpy(advs_np).float().to(DEVICE)
        returns = torch.from_numpy(returns_np).float().to(DEVICE)
        states, actions, old_logprobs = buffer.as_tensors()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        idxs = np.arange(len(returns_np))
        for _ in range(EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_idx = idxs[start:end]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advs = advs[mb_idx]

                new_logprobs, values, entropy = self.evaluate_actions(mb_states, mb_actions)
                ratio = (new_logprobs - mb_old_logprobs).exp()
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_returns)
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

    def save(self, path):
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

def evaluate(env, agent, num_episodes=10):
    returns = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            s = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = agent.model(s)
                action = torch.argmax(logits, dim=-1).item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    avg = float(np.mean(returns)) if returns else 0.0
    print(f"Evaluation over {num_episodes} episodes: Avg Reward {avg:.2f}")
    return avg

def train():
    name= ENV_NAME+'NoFrameskip-v4'
    env = AtariWrapper(gym.make(name))
    agent = PPOAgent(env.action_space.n)
    buffer = RolloutBuffer()

    for episode in range(MAX_EPISODES):
        state = env.reset()
        total_reward = 0.0
        done = False
        buffer.clear()

        while not done:
            action, logprob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, logprob, reward, done, value)
            state = next_state
            total_reward += reward

        agent.update(buffer)
        print(f"Episode {episode}, Reward {total_reward}")

        if episode % 50 == 0:
            evaluate(env, agent)
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/ppo_{ENV_NAME}_{episode}.pth")

    env.close()
    agent.save(f"models/ppo_{ENV_NAME}_final.pth")

def PPO_test(name):
    name= name+'NoFrameskip-v4'
    env = AtariWrapper(gym.make(name))
    env.seed()
    agent = PPOAgent(env.action_space.n)
    agent.load('../models/ppo_'+ENV_NAME+'_10600.pth')
    agent.model.eval()
    evaluate(env, agent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='Seaquest')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_eps', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--frame_stack', type=int, default=4)
    parser.add_argument('--max_episodes', type=int, default=13000)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    ENV_NAME = args.game
    GAMMA = args.gamma
    GAE_LAMBDA = args.gae_lambda
    CLIP_EPS = args.clip_eps
    LR = args.lr
    ENTROPY_COEF = args.entropy_coef
    VALUE_COEF = args.value_coef
    MAX_GRAD_NORM = args.max_grad_norm
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    FRAME_STACK = args.frame_stack
    MAX_EPISODES = args.max_episodes
    DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    # train()
    PPO_test(ENV_NAME)