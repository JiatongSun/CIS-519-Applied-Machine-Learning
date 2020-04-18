import numpy as np
import torch
from args import Args
from load_expert_data import load_initial_data, process_individual_observation
from generate_expert_trajectories import get_expert_action

def execute_dagger(args):
    env = args.env
    Q = args.expert_Q
    discretization = args.discretization
    model = args.model
    
    model.eval()
    
    num_observation = env.observation_space.shape[0]
    
    imitation_observations = np.zeros((1,num_observation),dtype=np.float)
    expert_actions = np.zeros((1,1),dtype=np.float)
    
    done = False  
    state = env.reset()
        
    while done != True: 
        observation = state
        data = process_individual_observation(args,observation)
        logit = model(data)
        real_action = torch.argmax(logit).item()
        expert_action = get_expert_action(Q,discretization,env,state) 
        
        imitation_observations = np.vstack((imitation_observations,
                                            state.reshape(1,-1)))
        expert_actions = np.vstack((expert_actions,
                                    np.array([expert_action])))
        
        next_state, _, done, _ = env.step(real_action) 
        state = next_state
    
    imitation_observations = np.delete(imitation_observations,0,axis=0)
    expert_actions = np.delete(expert_actions,0,axis=0)
    
    return imitation_observations, expert_actions

def aggregate_dataset(training_observations, training_actions, 
                      imitation_states, expert_actions):
    training_observations = np.vstack((training_observations,
                                       imitation_states))
    training_actions = np.vstack((training_actions, expert_actions))

    return training_observations, training_actions 

if __name__ == '__main__':
    args = Args()
    imitation_observations, expert_actions = execute_dagger(args)
    training_observations, training_actions = load_initial_data(args)
    training_observations, training_actions = \
        aggregate_dataset(training_observations, training_actions, 
                          imitation_observations, expert_actions)