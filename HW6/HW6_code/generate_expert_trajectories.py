import numpy as np
import os

def discretize(state, discretization, env):

    env_minimum = env.observation_space.low
    state_adj = (state - env_minimum)*discretization
    discretized_state = np.round(state_adj, 0).astype(int)

    return discretized_state

def get_expert_action(Q,discretization,env,state):
    state_disc = discretize(state,discretization,env)
    action = np.argmax(Q[state_disc[0], state_disc[1]])
    
    return action

def generate_expert_trajectories(Q, discretization, env, num_episodes=150, 
                                 data_path='./data'):
    total_samples = 0
    for i in range(num_episodes):
        episode_dict = {}
        episode_observations = []
        episode_actions = []
        
        done = False
        
        state = env.reset()
        
        while done != True:                 
            action = get_expert_action(Q,discretization,env,state)  
            episode_observations.append(state)
            episode_actions.append(action)  
            total_samples += 1                     
            next_state, _, done, _ = env.step(action) 
            state = next_state
        episode_dict['observations'] = episode_observations
        episode_dict['actions'] = episode_actions
        
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        np.savez_compressed(data_path+'/episode_number'+'_'+str(i)+'.npz',
                            **episode_dict)

    return total_samples 