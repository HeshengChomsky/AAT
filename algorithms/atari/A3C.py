import os
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
from Atarienv import AtariWrapper
import argparse

class CNNActorCritic(nn.Module):
    def __init__(self, action_dim, in_channels=4):
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

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()

def compute_returns_advantages(rewards, dones, values, gamma):
    R = 0.0 if dones[-1] else values[-1]
    returns = []
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1.0 - d)
        returns.append(R)
    returns = list(reversed(returns))
    advantages = [ret - val for ret, val in zip(returns, values[:-1])]
    return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)

def worker(rank, global_model, optimizer, env_name, gamma, value_coef, entropy_coef, t_max, max_episodes, save_path, seed, device):
    torch.manual_seed(seed + rank)
    env = AtariWrapper(gym.make(env_name))
    action_dim = env.action_space.n
    local_model = CNNActorCritic(action_dim).to(device)
    local_model.load_state_dict(global_model.state_dict())
    episode_count = 0
    while episode_count < max_episodes:
        state = env.reset()
        done = False
        rewards = []
        dones = []
        logprobs = []
        values = []
        entropies = []
        states_buf = []
        while not done and len(rewards) < t_max:
            s = torch.from_numpy(state).unsqueeze(0).to(device)
            logits, value = local_model(s)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action).squeeze(0)
            entropy = dist.entropy().mean()
            next_state, reward, done, _ = env.step(int(action.item()))
            rewards.append(float(reward))
            dones.append(float(done))
            logprobs.append(logprob)
            values.append(value.squeeze(0))
            entropies.append(entropy)
            states_buf.append(s)
            state = next_state
        with torch.no_grad():
            s_last = torch.from_numpy(state).unsqueeze(0).to(device)
            _, v_last = local_model(s_last)
        values.append(v_last.squeeze(0))
        returns, advantages = compute_returns_advantages(rewards, dones, values, gamma)
        logprobs_t = torch.stack(logprobs).to(device)
        values_t = torch.stack(values[:-1]).to(device)
        entropies_t = torch.stack(entropies).to(device)
        policy_loss = -(logprobs_t * advantages.to(device)).mean()
        value_loss = F.mse_loss(values_t, returns.to(device))
        entropy_loss = entropies_t.mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            if local_param.grad is not None:
                global_param._grad = local_param.grad
        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())
        episode_count += 1
        if rank == 0 and save_path and (episode_count % 50 == 0):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(global_model.state_dict(), save_path)
    env.close()

def evaluate(env_name, ckpt_path, episodes, device):
    env = AtariWrapper(gym.make(env_name))
    action_dim = env.action_space.n
    model = CNNActorCritic(action_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    returns = []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        done = False
        while not done:
            s = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(s)
                action = torch.argmax(logits, dim=-1).item()
            state, reward, done, _ = env.step(action)
            total += reward
        returns.append(total)
    env.close()
    return float(np.mean(returns)) if returns else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='Pong')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--t_max', type=int, default=20)
    parser.add_argument('--max_episodes', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='models/a3c_pong.pth')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--ckpt', type=str, default='models/a3c_pong.pth')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.cuda}' if args.cuda >= 0 and torch.cuda.is_available() else 'cpu')
    mp.set_start_method('spawn', force=True)
    env_name = args.game + 'NoFrameskip-v4'
    dummy_env = AtariWrapper(gym.make(env_name))
    action_dim = dummy_env.action_space.n
    dummy_env.close()
    global_model = CNNActorCritic(action_dim).to(device)
    global_model.share_memory()
    if args.mode == 'train':
        optimizer = SharedAdam(global_model.parameters(), lr=args.lr)
        processes = []
        for rank in range(args.num_workers):
            p = mp.Process(target=worker, args=(rank, global_model, optimizer, env_name, args.gamma, args.value_coef, args.entropy_coef, args.t_max, args.max_episodes, args.save_path, args.seed, device))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        avg = evaluate(env_name, args.ckpt, args.eval_episodes, device)
        print(f'Avg Reward {avg:.2f}')

if __name__ == '__main__':
    main()