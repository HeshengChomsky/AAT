import random
import numpy as np
import collections
import gym
import cv2
from collections import deque


# =============================
# Atari Wrapper
# =============================

class AtariWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, frame_stack=4):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(frame_stack, height, width), dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return obs

    def reset(self):
        obs = self.env.reset()[0]
        gray = self.observation(obs)
        for _ in range(self.frame_stack):
            self.frames.append(gray)
        return np.array(self.frames)

    def step(self, action):
        # obs, reward, terminated, truncated, info = self.env.step(action)
        # done = terminated or truncated
        step_result = self.env.step(action)
        if len(step_result) == 5: 
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        gray = self.observation(obs)
        self.frames.append(gray)
        return np.array(self.frames), reward, done, info
