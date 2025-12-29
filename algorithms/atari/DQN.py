import random
import numpy as np
import collections
import gym
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from algorithms.atari.Atarienv import AtariWrapper
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Hyperparameters
# =============================
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
BUFFER_SIZE = 100000
TARGET_UPDATE = 1000
N_STEP = 3
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = 100000
PER_EPS = 1e-6
FRAME_STACK = 4
MAX_EPISODES = 13000

# =============================
# Noisy Linear
# =============================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        eps_in = torch.randn(self.in_features)
        eps_out = torch.randn(self.out_features)
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

# =============================
# Dueling CNN Q Network
# =============================
class RainbowNetAtari(nn.Module):
    def __init__(self, action_dim, input_shape=(FRAME_STACK, 84, 84)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_value = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )
        self.fc_adv = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, action_dim)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255.0  # normalize
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        v = self.fc_value(x)
        a = self.fc_adv(x)
        return v + a - a.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

# =============================
# Prioritized Replay Buffer
# =============================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        max_prio = max(max_prio, self.eps)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        prios = self.priorities[:len(self.buffer)]
        prios = np.clip(prios, self.eps, None)

        probs = prios ** self.alpha
        probs /= probs.sum() + 1e-8

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities) + self.eps
        priorities = np.clip(priorities, self.eps, 1e6)
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# =============================
# Agent
# =============================
class RainbowAgent:
    def __init__(self, action_dim, input_shape=(FRAME_STACK, 84, 84)):
        self.q_net = RainbowNetAtari(action_dim, input_shape).to(DEVICE)
        self.target_net = RainbowNetAtari(action_dim, input_shape).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, PER_ALPHA, PER_EPS)

        self.n_step_buffer = deque(maxlen=N_STEP)
        self.action_dim = action_dim
        self.frame = 1

    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.q_net(state).argmax(1).item()

    def push(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < N_STEP:
            return

        R, next_s, done = 0, None, False
        for idx, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            R += (GAMMA ** idx) * r
            next_s, done = ns, d
            if d:
                break

        s, a = self.n_step_buffer[0][:2]
        self.memory.push((s, a, R, next_s, float(done)))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        beta = min(1.0, PER_BETA_START + self.frame * (1 - PER_BETA_START) / PER_BETA_FRAMES)
        samples, indices, weights = self.memory.sample(BATCH_SIZE, beta)

        states, actions, rewards, next_states, dones = map(
            np.array, zip(*samples)
        )

        states = torch.from_numpy(states).float().to(DEVICE)
        actions = torch.from_numpy(actions).long().to(DEVICE)
        rewards = torch.from_numpy(rewards).float().to(DEVICE)
        next_states = torch.from_numpy(next_states).float().to(DEVICE)
        dones = torch.from_numpy(dones).float().to(DEVICE)

        q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + (1 - dones) * (GAMMA ** N_STEP) * next_q

        td_error = target - q
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, td_error.detach().cpu().numpy())
        self.q_net.reset_noise()
        self.target_net.reset_noise()

        if self.frame % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.frame += 1

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'frame': self.frame,
            'action_dim': self.action_dim,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.frame = checkpoint.get('frame', 1)
        self.action_dim = checkpoint.get('action_dim', self.action_dim)


# =============================
# Evaluation
# =============================
def evaluation(env, agent, num_episodes=5):
    returns = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    avg = float(np.mean(returns)) if returns else 0.0
    print(f"Evaluation over {num_episodes} episodes: Avg Reward {avg:.2f}")
    return avg

# =============================
# Training
# =============================
def train():
    env = AtariWrapper(gym.make("PongNoFrameskip-v4"))
    agent = RainbowAgent(env.action_space.n)

    for episode in range(MAX_EPISODES):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.push((state, action, reward, next_state, done))
            agent.learn()

            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {episode}, Reward {total_reward}")
                break

        if episode % 50 == 0:
            evaluation(env, agent)
            agent.save(f"models/rainbow_pong_{episode}.pth")
            

    env.close()
    agent.save("models/rainbow_pong_final.pth")

if __name__ == "__main__":
    train()
