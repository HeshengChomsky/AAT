import os
import numpy as np
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from dm_control import suite
from dm_control.suite.wrappers import pixels

ENV_NAME = 'hopper'
TASK_NAME = 'stand'
IMAGE_SIZE = 84
STACK_FRAMES = 4
ACTION_REPEAT = 4
SEED = 42

MAX_EPISODES = 5000
MAX_STEPS = 1000 // ACTION_REPEAT
GAMMA = 0.99
LR = 1e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
UPDATE_ITER = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 1. 自定义 SharedAdam 优化器 ---
class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()


class PixelWrapper:
    def __init__(self, env, image_size=84, stack=4, action_repeat=4):
        self._env = env
        self.image_size = image_size
        self.stack = stack
        self.action_repeat = action_repeat
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])
        self.frames = deque(maxlen=stack)

    def reset(self):
        ts = self._env.reset()
        obs = self._get_pixel_obs(ts)
        for _ in range(self.stack):
            self.frames.append(obs)
        return self._stack_frames()

    def step(self, action):
        reward = 0.0
        for _ in range(self.action_repeat):
            ts = self._env.step(action)
            reward += (ts.reward or 0.0)
            if ts.last():
                break
        obs = self._get_pixel_obs(ts)
        self.frames.append(obs)
        return self._stack_frames(), reward, ts.last(), {}

    def _get_pixel_obs(self, ts):
        img = ts.observation['pixels']
        img = (img * 255).astype(np.uint8)
        return self.transform(img)

    def _stack_frames(self):
        return torch.cat(list(self.frames), dim=0)

    @property
    def action_spec(self):
        return self._env.action_spec()


class ConvEncoder(nn.Module):
    def __init__(self, stack=STACK_FRAMES, hidden=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(stack, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden), nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class A3CActorCritic(nn.Module):
    def __init__(self, action_size, hidden=512):
        super().__init__()
        self.encoder = ConvEncoder(STACK_FRAMES, hidden)
        self.mu = nn.Linear(hidden, action_size)
        self.sigma = nn.Linear(hidden, action_size)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        feat = self.encoder(x)
        mu = torch.tanh(self.mu(feat))
        sigma = F.softplus(self.sigma(feat)) + 1e-5
        v = self.value(feat).squeeze(1)
        return mu, sigma, v


class A3CAgent:
    def __init__(self, action_size):
        self.model = A3CActorCritic(action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.states, self.actions, self.rewards, self.dones = [], [], [], []

    @torch.no_grad()
    def act(self, state):
        s = state.unsqueeze(0).to(DEVICE)
        mu, sigma, _ = self.model(s)
        dist = torch.distributions.Normal(mu, sigma)
        a = dist.sample().clamp(-1, 1)
        return a.squeeze(0).cpu().numpy()

    def add_transition(self, state, action, reward, done):
        self.states.append(state.cpu())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(float(done))

    def update(self, next_state, done):
        with torch.no_grad():
            s_last = next_state.unsqueeze(0).to(DEVICE)
            _, _, v_last = self.model(s_last)
            R = 0.0 if done else float(v_last.item())
        returns = []
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + GAMMA * R * (1.0 - d)
            returns.append(R)
        returns.reverse()
        states = torch.stack(self.states).to(DEVICE)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32).to(DEVICE)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        mu, sigma, values = self.model(states)
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1).mean()
        adv = returns_t - values
        policy_loss = -(log_prob * adv.detach()).mean() - ENTROPY_COEF * entropy
        value_loss = F.mse_loss(values, returns_t)
        loss = policy_loss + VALUE_COEF * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
        self.optimizer.step()
        self.states, self.actions, self.rewards, self.dones = [], [], [], []

    def save(self, path, episode):
        os.makedirs(path, exist_ok=True)
        torch.save({'model': self.model.state_dict(), 'optim': self.optimizer.state_dict(), 'episode': episode}, os.path.join(path, f'ckpt_ep{episode}.pth'))

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optim'])
        return ckpt.get('episode', None)




if __name__ == '__main__':
    log_dir = 'runs/a3c_' + datetime.now().strftime('%m%d_%H%M')
    writer = SummaryWriter(log_dir)

    env_raw = suite.load(ENV_NAME, TASK_NAME)
    env_pixels = pixels.Wrapper(env_raw, pixels_only=True)
    env = PixelWrapper(env_pixels, IMAGE_SIZE, STACK_FRAMES, ACTION_REPEAT)

    action_spec = env.action_spec
    agent = A3CAgent(action_spec.shape[0])

    def map_action(a):
        low, high = action_spec.minimum, action_spec.maximum
        return low + (a + 1.0) * 0.5 * (high - low)

    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0.0
        for step in range(MAX_STEPS):
            action = agent.act(state)
            env_action = map_action(action)
            next_state, reward, done, _ = env.step(env_action)
            agent.add_transition(state, action, reward, done)
            state = next_state
            episode_reward += reward
            if len(agent.states) >= UPDATE_ITER or done:
                agent.update(next_state, done)
            if done:
                break
        writer.add_scalar('Reward/Train', episode_reward, episode)
        print(f"Ep: {episode} | Reward: {episode_reward:.2f}")