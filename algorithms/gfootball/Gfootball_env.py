import gym
import numpy as np
import cv2
from gym import spaces
from collections import deque
import gfootball.env as football_env

cv2.ocl.setUseOpenCL(False)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=96, height=72, grayscale=True):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        num_colors = 1 if self._grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )

    def observation(self, obs):
        frame = obs
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c * k), dtype=np.uint8
        )

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=-1)


def create_gfootball_env(env_name: str,
                         width: int = 96,
                         height: int = 72,
                         frame_stack: int = 4,
                         grayscale: bool = True,
                         seed: int = None):
    env = football_env.create_environment(
        env_name=env_name,
        representation='pixels',
        stacked=False,
        with_checkpoints=False,
        render=False,
        channels_order='last',
    )
    if seed is not None:
        try:
            env.seed(seed)
        except Exception:
            pass
    env = WarpFrame(env, width=width, height=height, grayscale=grayscale)
    env = FrameStack(env, k=frame_stack)
    return env


def make_env_from_args(args):
    return create_gfootball_env(
        env_name=args.env_name,
        width=96,
        height=72,
        frame_stack=4,
        grayscale=True,
        seed=getattr(args, "seed", None),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="11_vs_11_easy_stochastic")
    args = parser.parse_args()
    env = make_env_from_args(args)
    state=env.reset()
    print(state.shape)

if __name__ == '__main__':
    main()