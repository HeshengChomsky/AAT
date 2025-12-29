import os
import numpy as np
from collections import deque
from datetime import datetime

import torch
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

def get_env(env_name):
    env_raw = suite.load(env_name, task_name='stand')
    env_pixels = pixels.Wrapper(env_raw, pixels_only=True)
    env = PixelWrapper(env_pixels, IMAGE_SIZE, STACK_FRAMES, ACTION_REPEAT)
    return env