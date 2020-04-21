import torch

from args import get_args
from load_expert_data import process_individual_observation
from save_and_load import load_model

def test_model(args, record_frames=False):
    frames = []
    env = args.env 
    model = args.model
    state = env.reset()

    model.eval()

    episode_reward = 0

    success = False
    done = False 

    while not done:

        observation = state

        data = process_individual_observation(args,observation)
        logit = model(data)
        action = torch.argmax(logit).item()

        if record_frames: 
            frames.append(env.render())

        next_state, reward, done, _ = env.step(action) 
        episode_reward += reward

        if done:    
            if next_state[0] >= 0.5:
                success = True
            final_position = next_state[0]
            return final_position,success, frames, episode_reward
        else:
            state = next_state

def get_average_performance(args, run_for=1000):

    final_positions = 0
    successes = 0
    rewards = 0
  
    for ep in range(run_for):
        pos, success, _, episode_rewards = test_model(args, record_frames=False)   #test imitation policy
        final_positions += pos 
        rewards += episode_rewards
        if success:
          successes += 1
        print('Running Episode: ',ep,' Success: ', success)
        average_final_positions = final_positions/(ep+1)
        average_success_rate = 100*(successes/(ep+1))
        average_episode_rewards = rewards/(ep+1)
    
    return average_final_positions, average_success_rate, average_episode_rewards 


if __name__ == '__main__':
    args = get_args()
    args.model = load_model(args.env)
    
    # # visualize
    # final_position, success, frames, episode_reward \
    #     = test_model(args, record_frames=args.record_frames)
    # args.env.close()
    
    # average performance
    final_pos, succ_rate, ep_rwds = get_average_performance(args)
    print('Average Final Position achieved by the Agent: ',final_pos)
    print('Average Success Rate achieved by the Agent: ',succ_rate)
    print('Average Episode Reward achieved by the Agent: ',ep_rwds)