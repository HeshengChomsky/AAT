"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from algorithms.atari.Atarienv import AtariWrapper
import gym
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None, delta=None,target_model=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if x.size(1) <= block_size//3 else x[:, -block_size//3:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:]
        if delta is not None:
            delta = delta if delta.size(1) <= block_size//3 else delta[:, -block_size//3:]
        logits, _ = model(x_cond, actions=actions, target_model=target_model, rtgs=rtgs, timesteps=timesteps, delta=delta, targets=None)
        # # pluck the logits at the final step and scale by temperature
        # logits = logits[:, -1, :] / temperature
        # # optionally crop probabilities to only the top k options
        # if top_k is not None:
        #     logits = top_k_logits(logits, top_k)
        # # apply softmax to convert to probabilities
        # probs = F.softmax(logits, dim=-1)
        # # sample from the distribution or take the most likely
        # if sample:
        #     ix = torch.multinomial(probs, num_samples=1)
        # else:
        #     _, ix = torch.topk(probs, k=1, dim=-1)
        # # append to the sequence and continue
        # # x = torch.cat((x, ix), dim=1)
        # x = ix

    return logits

def get_target(target_model,state,model_type=None):
    if len(state.shape)<=3:
        adv_state = torch.from_numpy(state).unsqueeze(0).to(device)
    else:
        adv_state = torch.from_numpy(state).to(device)
    if model_type=='DQN':
        with torch.no_grad():
            logits= target_model.q_net(adv_state)
            action = torch.argmax(logits, dim=-1).item()
    elif model_type=='D4PG':
        with torch.no_grad():
            output = target_model.actor(adv_state)
            output = output.clamp(-1, 1)
            return output.squeeze(0).cpu().numpy()
    else:
        with torch.no_grad():
            logits, _ = target_model.model(adv_state)
            action = torch.argmax(logits, dim=-1).item()
    return action

def get_action_infor(game):
    name = game + 'NoFrameskip-v4'
    env = AtariWrapper(gym.make(name))
    action_dim=env.env.action_space.n
    env.close()
    return action_dim