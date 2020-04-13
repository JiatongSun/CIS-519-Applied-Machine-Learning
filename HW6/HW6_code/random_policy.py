import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
from PIL import Image

class ResizeObservation(gym.Wrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.env = env
        self.shape = tuple(shape)

    def render(self):
        obs = self.env.render(mode = 'rgb_array')
        im = Image.fromarray(np.uint8(obs))
        im = im.resize(self.shape)
        return np.asarray(im)



def dummy_policy(env, num_episodes):
    frames = []
    mean_reward = 0
    total_reward = 0
    record = False
    for i_episode in range(num_episodes):
        if i_episode == num_episodes-1: record = True
        episode_reward = 0
        observation = env.reset()
        for t in range(100):
            im = env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if record is True:
                frames.append(im)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        total_reward += episode_reward
    mean_reward = total_reward / num_episodes;
    return mean_reward, frames

if __name__ == '__main__':
    resize_observation_shape = 100
    env = gym.make('MountainCar-v0')
    env = ResizeObservation(env, resize_observation_shape)
    try:
        rew, frames = dummy_policy(env, 10)
    except (KeyboardInterrupt, SystemExit):
        raise
    finally:
        env.close()