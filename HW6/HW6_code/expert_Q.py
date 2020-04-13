import gym
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from resize_observation import ResizeObservation
from generate_expert_trajectories import generate_expert_trajectories

def discretize(state, discretization, env):

    env_minimum = env.observation_space.low
    state_adj = (state - env_minimum)*discretization
    discretized_state = np.round(state_adj, 0).astype(int)

    return discretized_state


def choose_action(epsilon, Q, state, env):
    """    
    Choose an action according to an epsilon greedy strategy.
    Args:
        epsilon (float): the probability of choosing a random action
        Q (np.array): The Q value matrix, here it is 3D for the two 
                      observation states and action states
        state (Box(2,)): the observation state, here it is 
                         [position, velocity]
        env: the RL environment 
        
    Returns:
        action (int): the chosen action
    """
    action = 0
    if np.random.random() < 1 - epsilon:
        action = np.argmax(Q[state[0], state[1]]) 
    else:
        action = np.random.randint(0, env.action_space.n)
  
    return action


def update_epsilon(epsilon, decay_rate):
    """
    Decay epsilon by the specified rate.
    
    Args:
        epsilon (float): the probability of choosing a random action
        decay_rate (float): the decay rate (between 0 and 1) to scale 
                            epsilon by
        
    Returns:
        updated epsilon
    """
  
    epsilon *= decay_rate

    return epsilon


def update_Q(Q, state_disc, next_state_disc, action, discount, learning_rate, reward, terminal):
    """
    
    Update Q values following the Q-learning update rule. 
    
    Be sure to handle the terminal state case.
    
    Args:
        Q (np.array): The Q value matrix, here it is 3D for the two 
                      observation states and action states
        state_disc (np.array): the discretized version of the current 
                               observation state [position, velocity]
        next_state_disc (np.array): the discretized version of the next 
                                    observation state [position, velocity]
        action (int): the chosen action
        discount (float): the discount factor, may be referred 
                          to as gamma
        learning_rate (float): the learning rate, may be referred 
                               to as alpha
        reward (float): the current (immediate) reward
        terminal (bool): flag for whether the state is terminal
        
    Returns:
        Q, with the [state_disc[0], state_disc[1], action] entry updated.
    """     
    if terminal:        
        Q[state_disc[0], state_disc[1], action] = reward

    # Adjust Q value for current state
    else:
        delta = learning_rate*(reward + discount*np.max(Q[next_state_disc[0], next_state_disc[1]]) - Q[state_disc[0], state_disc[1],action])
        Q[state_disc[0], state_disc[1],action] += delta
  
    return Q


def Qlearning(Q, discretization, env, learning_rate, discount, 
              epsilon, decay_rate, max_episodes=5000):
    """
    
    The main Q-learning function, utilizing the functions implemented above.
          
    """
    reward_list = []
    position_list = []
    success_list = []
    success = 0 # count of number of successes reached 
    frames = []
  
    for i in range(max_episodes):
        # Initialize parameters
        done = False # indicates whether the episode is done
        terminal = False # indicates whether the episode is done AND 
                         # the car has reached the flag (>=0.5 position)
        tot_reward = 0 # sum of total reward over a single
        state = env.reset() # initial environment state
        state_disc = discretize(state,discretization,env)

        while done != True:                 
            # Determine next action 
            action = choose_action(epsilon, Q, state_disc, env)                                      
            # Get next_state, reward, and done using env.step(), 
            # see http://gym.openai.com/docs/#environments for reference
            if i==1 or i==(max_episodes-1):
              frames.append(env.render())
            next_state, reward, done, _ = env.step(action) 
            # Discretize next state 
            next_state_disc = discretize(next_state,discretization,env)
            # Update terminal
            terminal = done and next_state[0]>=0.5
            # Update Q
            Q = update_Q(Q,state_disc,next_state_disc,action,discount,
                         learning_rate, reward, terminal)  
            # Update tot_reward, state_disc, and success (if applicable)
            tot_reward += reward
            state_disc = next_state_disc

            if terminal: success +=1 
        
        #Update level of epsilon using update_epsilon()
        epsilon = update_epsilon(epsilon, decay_rate)

        # Track rewards
        reward_list.append(tot_reward)
        position_list.append(next_state[0])
        success_list.append(success/(i+1))

        if (i+1) % 100 == 0:
            print('Episode: ', i+1, 'Average Reward over 100 Episodes: ',
                  np.mean(reward_list))
            reward_list = []
                
    env.close()
    
    return Q, position_list, success_list, frames

if __name__ == '__main__':
    # Initialize Mountain Car Environment
    env = gym.make('MountainCar-v0')
    
    env = ResizeObservation(env,100) #Resize observations
    
    env.seed(42)
    np.random.seed(42)
    env.reset()
    
    # Parameters    
    learning_rate = 0.2 
    discount = 0.9
    epsilon = 0.8 
    decay_rate = 0.95
    max_episodes = 5000
    discretization = np.array([10,100])
    
    
    #InitQ
    num_states = (env.observation_space.high - 
                  env.observation_space.low) * discretization
    #Size of discretized state space 
    num_states = np.round(num_states, 0).astype(int) + 1
    # Initialize Q table
    Q = np.random.uniform(low = -1, 
                          high = 1, 
                          size = (num_states[0], 
                                  num_states[1], 
                                  env.action_space.n))
    
    try:
        # Run Q Learning by calling your Qlearning() function
        Q, position, successes, frames = Qlearning(Q, discretization, 
                                                   env, learning_rate, 
                                                   discount, epsilon, 
                                                   decay_rate, 
                                                   max_episodes)
    
        np.save('./expert_Q.npy',Q) #Save the expert
        
        plt.plot(successes)
        plt.xlabel('Episode')
        plt.ylabel('% of Episodes with Success')
        plt.title('% Successes')
        plt.show()
        plt.close()
        
        p = pd.Series(position)
        ma = p.rolling(3).mean()
        plt.plot(p, alpha=0.8)
        plt.plot(ma)
        plt.xlabel('Episode')
        plt.ylabel('Position')
        plt.title('Car Final Position')
        plt.show()
        
        num_episodes = 100
        data_path = './data'
        total_samples = generate_expert_trajectories(Q,discretization,
                                                     env,num_episodes,
                                                     data_path)
        
        print('Recorded ', total_samples, 'samples')
    except (KeyboardInterrupt, SystemExit):
        print("keyboard interrupt")
    finally:
        env.close()