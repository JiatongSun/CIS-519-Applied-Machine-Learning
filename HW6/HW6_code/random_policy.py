import gym
from resize_observation import ResizeObservation

def dummy_policy(env, num_episodes):
    frames = []
    mean_reward = 0
    total_reward = 0
    for i_episode in range(num_episodes):
        episode_reward = 0
        observation = env.reset()
        for t in range(200):
            im = env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if i_episode == num_episodes-1: 
                frames.append(im)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        total_reward += episode_reward
    mean_reward = total_reward / num_episodes
    print(f'mean reward: {mean_reward}')
    return mean_reward, frames

if __name__ == '__main__':
    resize_observation_shape = 100
    env = gym.make('MountainCar-v0')
    env = ResizeObservation(env, resize_observation_shape)
    try:
        rew, frames = dummy_policy(env, 10)
    except (KeyboardInterrupt, SystemExit):
        print("keyboard interrupt")
    finally:
        env.close()