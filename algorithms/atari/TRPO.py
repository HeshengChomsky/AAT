import os
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from algorithms.Atarienv import AtariWrapper
import argparse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ENV_NAME = 'Pong'
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_KL = 1e-2
DAMPING = 1e-3
VF_LR = 1e-3
CG_ITERS = 10
LS_BACKTRACKS = 10
LS_ACCEPT_RATIO = 0.1
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
        self.old_logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, old_logprob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.old_logprobs.append(old_logprob)
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
        old_logprobs = torch.from_numpy(np.array(self.old_logprobs)).float().to(DEVICE)
        return states, actions, old_logprobs

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.old_logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

class TRPOAgent:
    def __init__(self, action_dim):
        self.model = CNNActorCritic(action_dim).to(DEVICE)
        self.vf_optimizer = torch.optim.Adam(self.model.value.parameters(), lr=VF_LR)

    def select_action(self, state):
        s = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, value = self.model(s)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def flat_params(self):
        return torch.cat([p.view(-1) for p in self.model.parameters()])

    def set_params(self, flat_params):
        idx = 0
        for p in self.model.parameters():
            n = p.numel()
            p.data.copy_(flat_params[idx:idx + n].view_as(p))
            idx += n

    def flat_grad(self, grads):
        parts = []
        for p, g in zip(self.model.parameters(), grads):
            if g is None:
                parts.append(torch.zeros_like(p).view(-1))
            else:
                parts.append(g.contiguous().view(-1))
        return torch.cat(parts)

    def compute_kl(self, states, old_logits):
        new_logits, _ = self.model(states)
        old_probs = torch.softmax(old_logits.detach(), dim=1)
        new_log_probs = torch.log_softmax(new_logits, dim=1)
        old_log_probs = torch.log_softmax(old_logits.detach(), dim=1)
        kl = (old_probs * (old_log_probs - new_log_probs)).sum(dim=1).mean()
        return kl

    def fisher_vector_product(self, states, old_logits, v):
        kl = self.compute_kl(states, old_logits)
        grads = torch.autograd.grad(kl, self.model.parameters(), create_graph=True, allow_unused=True)
        flat_grad_kl = self.flat_grad(grads)
        kl_v = (flat_grad_kl * v).sum()
        grads2 = torch.autograd.grad(kl_v, self.model.parameters(), retain_graph=True, allow_unused=True)
        flat_grad2 = self.flat_grad(grads2)
        return flat_grad2 + DAMPING * v

    def conjugate_gradient(self, states, old_logits, b, iters=CG_ITERS, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rr = torch.dot(r, r)
        for _ in range(iters):
            Avp = self.fisher_vector_product(states, old_logits, p)
            alpha = rr / (torch.dot(p, Avp) + 1e-8)
            x += alpha * p
            r -= alpha * Avp
            rr_new = torch.dot(r, r)
            if rr_new < residual_tol:
                break
            beta = rr_new / (rr + 1e-8)
            p = r + beta * p
            rr = rr_new
        return x

    def linesearch(self, states, actions, old_logprobs, old_logits, fullstep, expected_improve_rate):
        f = lambda: self.surrogate_loss(states, actions, old_logprobs)
        prev_params = self.flat_params()
        prev_loss = f().item()
        for stepfrac in [0.5 ** i for i in range(LS_BACKTRACKS)]:
            new_params = prev_params + stepfrac * fullstep
            self.set_params(new_params)
            loss = f().item()
            kl = self.compute_kl(states, old_logits).item()
            actual_improve = prev_loss - loss
            if actual_improve > LS_ACCEPT_RATIO * expected_improve_rate and kl <= MAX_KL:
                return True
        self.set_params(prev_params)
        return False

    def surrogate_loss(self, states, actions, old_logprobs):
        logits, _ = self.model(states)
        dist = Categorical(logits=logits)
        new_logprobs = dist.log_prob(actions)
        return -(new_logprobs * self.advs).mean()

    def update(self, buffer):
        returns_np, advs_np = buffer.compute_returns_advantages(GAMMA, GAE_LAMBDA)
        advs = torch.from_numpy(advs_np).float().to(DEVICE)
        returns = torch.from_numpy(returns_np).float().to(DEVICE)
        states, actions, old_logprobs = buffer.as_tensors()
        logits, values = self.model(states)
        self.advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        dist = Categorical(logits=logits)
        new_logprobs = dist.log_prob(actions)
        ratio = torch.exp(new_logprobs - old_logprobs)
        loss = -(ratio * self.advs).mean()
        grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True, allow_unused=True)
        g = self.flat_grad(grads)
        step_dir = self.conjugate_gradient(states, logits.detach(), g)
        fvp = self.fisher_vector_product(states, logits.detach(), step_dir)
        shs = 0.5 * torch.dot(step_dir, fvp)
        lagrange_multiplier = torch.sqrt((2.0 * MAX_KL) / (shs + 1e-8))
        fullstep = lagrange_multiplier * step_dir
        expected_improve = (g * fullstep).sum().item()
        self.linesearch(states, actions, old_logprobs, logits.detach(), fullstep, expected_improve)
        for _ in range(5):
            self.vf_optimizer.zero_grad()
            _, values = self.model(states)
            vf_loss = F.mse_loss(values, returns)
            vf_loss.backward()
            self.vf_optimizer.step()
        buffer.clear()

    def save(self, path):
        torch.save({'model': self.model.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt['model'])

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
    name = ENV_NAME + 'NoFrameskip-v4'
    env = AtariWrapper(gym.make(name))
    agent = TRPOAgent(env.action_space.n)
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
            agent.save(f"models/trpo_{ENV_NAME}_{episode}.pth")
    env.close()
    agent.save(f"models/trpo_{ENV_NAME}_final.pth")

def TRPO_test(name):
    name = name + 'NoFrameskip-v4'
    env = AtariWrapper(gym.make(name))
    agent = TRPOAgent(env.action_space.n)
    agent.load(f'models/trpo_{ENV_NAME}_final.pth')
    agent.model.eval()
    evaluate(env, agent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='Pong')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--max_kl', type=float, default=1e-2)
    parser.add_argument('--damping', type=float, default=1e-3)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--ls_backtracks', type=int, default=10)
    parser.add_argument('--ls_accept_ratio', type=float, default=0.1)
    parser.add_argument('--frame_stack', type=int, default=4)
    parser.add_argument('--max_episodes', type=int, default=13000)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    ENV_NAME = args.game
    GAMMA = args.gamma
    GAE_LAMBDA = args.gae_lambda
    MAX_KL = args.max_kl
    DAMPING = args.damping
    VF_LR = args.vf_lr
    CG_ITERS = args.cg_iters
    LS_BACKTRACKS = args.ls_backtracks
    LS_ACCEPT_RATIO = args.ls_accept_ratio
    FRAME_STACK = args.frame_stack
    MAX_EPISODES = args.max_episodes
    DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    train()
    # TRPO_test(args.game)