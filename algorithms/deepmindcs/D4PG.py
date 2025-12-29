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

# ---------------- 超参数 ----------------
ENV_NAME = 'hopper'
TASK_NAME = 'stand'
IMAGE_SIZE = 84
STACK_FRAMES = 4
ACTION_REPEAT = 4
SEED = 42

MAX_EPISODES = 5000
MAX_STEPS = 1000 // ACTION_REPEAT
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100_000
WARMUP_STEPS = 1000
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
V_MIN, V_MAX = -150.0, 150.0
N_ATOMS = 51
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)
NOISE_STD = 0.2
NOISE_CLIP = 0.5
POLICY_DELAY = 1  # D4PG 通常每个 step 都更新 actor，也可设为 2
SAVE_EVERY_EPOCH = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- 环境包装 ----------------
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
            if ts.last(): break
        obs = self._get_pixel_obs(ts)
        self.frames.append(obs)
        return self._stack_frames(), reward, ts.last(), {}

    def _get_pixel_obs(self, ts):
        img = ts.observation['pixels']
        return self.transform(img)

    def _stack_frames(self):
        return torch.cat(list(self.frames), dim=0)

    @property
    def action_spec(self):
        return self._env.action_spec()

# ---------------- 网络结构 ----------------
class ConvEncoder(nn.Module):
    def __init__(self, stack=4, hidden=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(stack, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, hidden), nn.ReLU())

    def forward(self, x):
        return self.fc(self.conv(x))

class Actor(nn.Module):
    def __init__(self, encoder, action_size, hidden=512):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_size), nn.Tanh()
        )

    def forward(self, x):
        return self.fc(self.encoder(x))

class Critic(nn.Module):
    def __init__(self, encoder, action_size, n_atoms=N_ATOMS, hidden=512):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(hidden + action_size, hidden), nn.ReLU(),
            nn.Linear(hidden, n_atoms)
        )

    def forward(self, x, a):
        feat = self.encoder(x)
        return self.fc(torch.cat([feat, a], dim=1))

# ---------------- 经验回放 ----------------
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state.cpu(), action, reward, next_state.cpu(), done))

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.stack(state).to(DEVICE), 
                torch.tensor(np.array(action), dtype=torch.float32).to(DEVICE),
                torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(DEVICE),
                torch.stack(next_state).to(DEVICE),
                torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(DEVICE))

    def __len__(self):
        return len(self.buffer)

# ---------------- D4PG Agent ----------------
class D4PGAgent:
    def __init__(self, action_size, action_spec):
        self.action_size = action_size
        # 注意：不再在 act 内部强制转换，保持网络输出和 Buffer 动作一致
        self.actor = Actor(ConvEncoder(STACK_FRAMES), action_size).to(DEVICE)
        self.actor_target = Actor(ConvEncoder(STACK_FRAMES), action_size).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(ConvEncoder(STACK_FRAMES), action_size).to(DEVICE)
        self.critic_target = Critic(ConvEncoder(STACK_FRAMES), action_size).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optim_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.replay = ReplayBuffer()
        self.z = torch.linspace(V_MIN, V_MAX, N_ATOMS).to(DEVICE)
        self.step_count = 0

    @torch.no_grad()
    def act(self, state, noise=True):
        state = state.unsqueeze(0).to(DEVICE)
        action = self.actor(state)
        if noise:
            action += torch.randn_like(action) * NOISE_STD
        action = action.clamp(-1, 1)
        return action.squeeze(0).cpu().numpy()

    def update(self):
        if len(self.replay) < BATCH_SIZE: return
        self.step_count += 1
        state, action, reward, next_state, done = self.replay.sample(BATCH_SIZE)

        # --- 1. 更新 Critic (Categorical Projection) ---
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            # 目标动作也保持在 [-1, 1]
            next_dist = self.critic_target(next_state, next_action)
            next_prob = F.softmax(next_dist, dim=1)

            # 投影计算
            target_z = reward + (1 - done) * GAMMA * self.z.unsqueeze(0)
            target_z = target_z.clamp(V_MIN, V_MAX)
            
            b = (target_z - V_MIN) / DELTA_Z
            l = b.floor().long().clamp(0, N_ATOMS - 1)
            u = b.ceil().long().clamp(0, N_ATOMS - 1)
            
            # 这里的数值修正是为了解决 floor == ceil 的情况
            u[(l == u) & (u < N_ATOMS - 1)] += 1
            l[(l == u) & (l > 0)] -= 1

            m = torch.zeros(BATCH_SIZE, N_ATOMS).to(DEVICE)
            offset = torch.linspace(0, (BATCH_SIZE - 1) * N_ATOMS, BATCH_SIZE).long().unsqueeze(1).to(DEVICE)
            
            # 使用修正后的索引进行 index_add
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_prob * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_prob * (b - l.float())).view(-1))

        log_dist = F.log_softmax(self.critic(state, action), dim=1)
        critic_loss = -(m * log_dist).sum(1).mean()

        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        # --- 2. 更新 Actor ---
        if self.step_count % POLICY_DELAY == 0:
            # 此时 actor_action 是 Tanh 输出 [-1, 1]，直接喂给 critic
            actor_action = self.actor(state)
            dist = self.critic(state, actor_action)
            prob = F.softmax(dist, dim=1)
            q_expected = (prob * self.z.unsqueeze(0)).sum(1)
            actor_loss = -q_expected.mean()

            self.optim_actor.zero_grad()
            actor_loss.backward()
            self.optim_actor.step()

            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

def evaluate(env, agent, episodes=10, noise=False):
    action_spec = env.action_spec
    returns = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, noise=noise)
            env_action = action_spec.minimum + (action + 1.0) * 0.5 * (action_spec.maximum - action_spec.minimum)
            state, reward, done, _ = env.step(env_action)
            total_reward += reward
        returns.append(total_reward)
    avg = float(np.mean(returns)) if returns else 0.0
    print(f"Evaluation over {episodes} episodes: Avg Reward {avg:.2f}")
    return avg

def main():
    # 路径与初始化
    log_dir = 'runs/d4pg_' + datetime.now().strftime('%m%d_%H%M')
    writer = SummaryWriter(log_dir)
    
    env_raw = suite.load(ENV_NAME, TASK_NAME)
    env_pixels = pixels.Wrapper(env_raw, pixels_only=True)
    env = PixelWrapper(env_pixels, IMAGE_SIZE, STACK_FRAMES, ACTION_REPEAT)
    
    action_spec = env.action_spec
    agent = D4PGAgent(action_spec.shape[0], action_spec)
    
    # 映射函数：从 [-1, 1] 映射到环境真实范围
    def map_action(a):
        low, high = action_spec.minimum, action_spec.maximum
        return low + (a + 1.0) * 0.5 * (high - low)

    global_step = 0
    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            # 获取归一化动作 [-1, 1]
            action = agent.act(state)
            
            # 映射后执行，但存入 buffer 的是原始 action
            env_action = map_action(action)
            next_state, reward, done, _ = env.step(env_action)
            
            agent.replay.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            global_step += 1
            
            if global_step > WARMUP_STEPS:
                agent.update()
            if done: break
            
        writer.add_scalar('Reward/Train', episode_reward, episode)
        print(f"Ep: {episode} | Reward: {episode_reward:.2f} | Steps: {global_step}")

if __name__ == '__main__':
    main()