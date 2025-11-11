import gym
# import gym_pygame

#!pip install git+https://github.com/ntasfi/PyGame-Learning-Environment.git
#!pip install git+https://github.com/qlan3/gym-games.git


#定义环境
class MyWrapper(gym.Wrapper):

    def __init__(self):
        env = gym.make('Pong-PLE-v0')
        super().__init__(env)
        self.env = env
        self.step_n = 0

    def reset(self):
        state = self.env.reset()
        self.step_n = 0
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.step_n += 1
        if self.step_n >= 400:
            done = True
        return state, reward, done, info


env = MyWrapper()

env.reset()

#认识游戏环境
def test_env():
    print('env.observation_space=', env.observation_space)
    print('env.action_space=', env.action_space)

    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)

    print('state=', state)
    print('action=', action)
    print('next_state=', next_state)
    print('reward=', reward)
    print('done=', done)
    print('info=', info)


test_env()

from matplotlib import pyplot as plt

# matplotlib inline


#打印游戏
def show():
    plt.figure(figsize=(3, 3))
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()


show()