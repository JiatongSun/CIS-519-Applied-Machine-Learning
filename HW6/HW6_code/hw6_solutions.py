import numpy as np
from PIL import Image
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
from torch import save
from torch import load
import copy
from os import path
import glob

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

def discretize(state, discretization, env):
    env_minimum = env.observation_space.low
    state_adj = (state - env_minimum)*discretization
    discretized_state = np.round(state_adj, 0).astype(int)

    return discretized_state


def choose_action(epsilon, Q, state, env):
    action = 0
    if np.random.random() < 1 - epsilon:
        action = np.argmax(Q[state[0], state[1]]) 
    else:
        action = np.random.randint(0, env.action_space.n)
  
    return action


def update_epsilon(epsilon, decay_rate):
    epsilon *= decay_rate

    return epsilon


def update_Q(Q, state_disc, next_state_disc, action, discount, learning_rate, reward, terminal):
    if terminal:        
        Q[state_disc[0], state_disc[1], action] = reward

    # Adjust Q value for current state
    else:
        delta = learning_rate*(reward + discount*np.max(Q[next_state_disc[0], next_state_disc[1]]) - Q[state_disc[0], state_disc[1],action])
        Q[state_disc[0], state_disc[1],action] += delta
  
    return Q

def Qlearning(Q, discretization, env, learning_rate, discount, epsilon, decay_rate, max_episodes=5000):
    reward_list = []
    position_list = []
    success_list = []
    success = 0 # count of number of successes reached 
    frames = []
  
    for i in range(max_episodes):
        # Initialize parameters
        done = False # indicates whether the episode is done
        terminal = False # indicates whether the episode is done AND the car has reached the flag (>=0.5 position)
        tot_reward = 0 # sum of total reward over a single
        state = env.reset() # initial environment state
        state_disc = discretize(state,discretization,env)

        while done != True:                 
            # Determine next action 
            action = choose_action(epsilon, Q, state_disc, env)                                      
            # Get next_state, reward, and done using env.step(), see http://gym.openai.com/docs/#environments for reference
            if i==1 or i==(max_episodes-1):
              frames.append(env.render())
            next_state, reward, done, _ = env.step(action) 
            # Discretize next state 
            next_state_disc = discretize(next_state,discretization,env)
            # Update terminal
            terminal = done and next_state[0]>=0.5
            # Update Q
            Q = update_Q(Q,state_disc,next_state_disc,action,discount,learning_rate, reward, terminal)  
            # Update tot_reward, state_disc, and success (if applicable)
            tot_reward += reward
            state_disc = next_state_disc

            if terminal: success +=1 
            
        epsilon = update_epsilon(epsilon, decay_rate) #Update level of epsilon using update_epsilon()

        # Track rewards
        reward_list.append(tot_reward)
        position_list.append(next_state[0])
        success_list.append(success/(i+1))

        if (i+1) % 100 == 0:
            print('Episode: ', i+1, 'Average Reward over 100 Episodes: ',np.mean(reward_list))
            reward_list = []
                
    env.close()
    
    return Q, position_list, success_list, frames

def get_expert_action(Q,discretization,env,state):
    state_disc = discretize(state,discretization,env)
    action = np.argmax(Q[state_disc[0], state_disc[1]])

    return action

def generate_expert_trajectories(Q, discretization, env, num_episodes=150, data_path='./data'):
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
        
        import os
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        np.savez_compressed(data_path+'/episode_number'+'_'+str(i)+'.npz',
                            **episode_dict)

    return total_samples

def load_initial_data(args):
    num_file = 0
    for file in glob.glob(path.join(args.datapath, '*.npz')):
        if num_file >= args.initial_episodes_to_use: break
        data = np.load(file)
        data_len = len(data['observations'])
        cur_observations = data['observations'].reshape(data_len,-1)
        cur_actions = data['actions'].reshape(data_len,-1)
        if num_file == 0:
            training_observations = cur_observations.copy()
            training_actions = cur_actions.copy()
        else:
            training_observations = np.vstack((training_observations, 
                                               cur_observations))
            training_actions = np.vstack((training_actions,
                                          cur_actions))
        num_file += 1
    training_actions = training_actions.reshape(-1)
    
    return training_observations, training_actions

def load_dataset(args, observations, actions, batch_size=64, data_transforms=None, num_workers=0):
    class dataset(Dataset):
        def __init__(self,transform,observations,actions):
            self.observations = observations
            self.actions = actions
            self.transform = transform
    
        def __len__(self):
            return len(self.observations)
    
        def __getitem__(self,idx):
            observations = self.transform(self.observations).float()
            actions = torch.tensor(self.actions.reshape(-1))
            item = {'observations': observations[:,idx], 
                    'actions': actions[idx]}
            return item
        
    if data_transforms is None:
        transform = args.transform
    else:
        transform = data_transforms
    
    train_dataset = dataset(transform,observations,actions)
    
    dataloader = DataLoader(train_dataset, 
                            num_workers=num_workers, 
                            batch_size=batch_size)
    return dataloader

def process_individual_observation(args,observation):
    data = args.transform(observation.reshape(1,-1)).float()
    return data

class StatesNetwork(nn.Module):
    def __init__(self, env):
        super(StatesNetwork,self).__init__()
        num_observation = env.observation_space.shape[0]
        num_action = env.action_space.n
        self.fc = nn.Sequential(
            nn.Linear(num_observation,50),
            nn.ReLU(True),
            nn.Linear(50,num_action)
        )
    
    def forward(self, x):    
        x = x.view(x.size(0),-1)
        forward_pass = self.fc(x)

        return forward_pass

def train_model(args):
    # initialization
    model = args.model
    epoch = args.num_epochs
    optimizer = args.optimizer
    criterion = args.criterion
    training_observations, training_actions = load_initial_data(args)
    
    dataset_sizes = len(training_observations)
    dataloader = load_dataset(args,
                              training_observations,
                              training_actions, 
                              args.batch_size,
                              args.transform)
    num_valid_episode = args.num_valid_episode
        
    global_step = 0
    
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_flag = True
    
    for e in range(epoch):
        if train_flag is False:
            break
        
        # train
        running_loss = 0.0
        running_corrects = 0
        for i,batch in enumerate(dataloader):
            inputs, labels = batch['observations'], batch['actions']
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, axis=1)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            global_step += 1
        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_corrects.double() / dataset_sizes
        print('epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(e,
                                                          epoch_loss, 
                                                          epoch_acc))
        
        # valid
        total_reward = 0
        total_success = 0
        for i_episode in range(num_valid_episode):
            final_position, success, frames, episode_reward \
                = test_model(args, record_frames=args.record_frames)
            total_success += success
            total_reward += episode_reward

        success_rate = total_success / num_valid_episode
        mean_reward = total_reward / num_valid_episode
        print('success rate: {} mean reward: {}\n'.format(success_rate, 
                                                          mean_reward))
        
        if epoch_acc > best_acc:
            best_epoch = epoch
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
                
        if epoch-best_epoch > 10:
            print('early stopping!')
            train_flag = False

    print('Best Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    save_model(model)

    return model

def save_model(model):
    return save(model.state_dict(), 'states_net.th')
    
def load_model(env):
    r = StatesNetwork(env)
    r.load_state_dict(load('states_net.th', map_location='cpu'))
    return r

def execute_dagger(args):
    env = args.env
    Q = args.Q
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
        
        imitation_observations = np.vstack((imitation_observations, state.reshape(1,-1)))
        expert_actions = np.vstack((expert_actions, np.array([expert_action])))
        
        next_state, _, done, _ = env.step(real_action) 
        state = next_state
    
    imitation_observations = np.delete(imitation_observations,0,axis=0)
    expert_actions = np.delete(expert_actions,0,axis=0).reshape(-1)
    
    return imitation_observations, expert_actions

def aggregate_dataset(training_observations, training_actions, imitation_states, expert_actions):
    training_observations = np.vstack((training_observations, imitation_states))
    training_actions = np.hstack((training_actions, expert_actions))

    return training_observations, training_actions

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

def predict(model, inputs, device='cpu'):
    inputs = inputs.to(device)
    logits = model(inputs)
    return F.softmax(logits, -1)

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

def imitate(args):
    model = args.model
    epoch = args.num_epochs
    optimizer = args.optimizer
    criterion = args.criterion
    training_observations, training_actions = load_initial_data(args)
    
    dataset_sizes = len(training_observations)
    dataloader = load_dataset(args,
                              training_observations,
                              training_actions, 
                              args.batch_size,
                              args.transform)
    num_valid_episode = args.num_valid_episode
        
    global_step = 0
    
    final_positions = []
    success_history = []
    frames = []
    reward_history = []
    
    for dagger_iter in range(args.max_dagger_iterations):
        print('\ndagger_iter: {}'.format(dagger_iter+1))
        for e in range(epoch):
            
            # train
            running_loss = 0.0
            running_corrects = 0
            for i,batch in enumerate(dataloader):
                inputs, labels = batch['observations'], batch['actions']
                labels = labels.long()
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, axis=1)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                global_step += 1

            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects.double() / dataset_sizes
            print('epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(e+1,
                                                              epoch_loss, 
                                                              epoch_acc))
            
            # valid
            total_reward = 0
            total_success = 0
            for i_episode in range(num_valid_episode):
                final_position, success, _, episode_reward \
                    = test_model(args)
                total_success += success
                total_reward += episode_reward
    
            success_rate = total_success / num_valid_episode
            mean_reward = total_reward / num_valid_episode
            print('success rate: {} mean reward: {}'.format(success_rate, 
                                                            mean_reward))
        
        # update args
        args.model.load_state_dict(model.state_dict())
        
        # dataset aggregation
        if args.do_dagger is True:
            args.model.load_state_dict(model.state_dict())
            imitation_observations, expert_actions = execute_dagger(args)
            training_observations, training_actions = \
                aggregate_dataset(training_observations, training_actions, 
                                  imitation_observations, expert_actions)
            dataset_sizes = len(training_observations)
            dataloader = load_dataset(args,
                                      training_observations,
                                      training_actions,
                                      args.batch_size,
                                      args.transform)
            final_position, success, frames_ep, episode_reward \
                = test_model(args, record_frames=args.record_frames)
            final_positions.append(final_position)
            success_history.append(success)
            frames.append(frames_ep)
            reward_history.append(episode_reward)

    return final_positions, success_history, frames, reward_history, args

class Args(object):
    def  __init__(self):
        # model
        self.env = ResizeObservation(gym.make('MountainCar-v0'), 100)
        self.model = StatesNetwork(self.env)
        self.transform = tf.Compose([tf.ToTensor()])
        
        # data & log
        self.datapath = './data'
        self.log_dir = './logdir'
        
        # expert
        self.Q = np.load('./expert_Q.npy')
        self.discretization = np.array([10,100])
        
        # train
        self.initial_episodes_to_use = 20
        self.batch_size = 128
        self.lr = 0.1
        self.num_epochs = 2
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.lr)
        self.criterion = torch.nn.CrossEntropyLoss() 
        
        # dagger
        self.do_dagger = False
        if self.do_dagger:
            self.max_dagger_iterations = 18
        else:
            self.max_dagger_iterations = 1
        
        # valid
        self.record_frames = True
        self.num_valid_episode = 5
    
def get_args():
    args = Args()
    return args

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
    # print('Running Episode: ',ep,' Success: ', success)
    average_final_positions = final_positions/(ep+1)
    average_success_rate = 100*(successes/(ep+1))
    average_episode_rewards = rewards/(ep+1)

  return average_final_positions, average_success_rate, average_episode_rewards 

if __name__ == '__main__':
    args = get_args()
    final_positions, success_history, frames, reward_history, args = imitate(args)