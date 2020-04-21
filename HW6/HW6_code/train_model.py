import torch
import copy
import torch.utils.tensorboard as tb
from os import path

from args import get_args
from load_expert_data import load_initial_data, load_dataset
from save_and_load import save_model, load_model
from test_model import test_model
from dagger import execute_dagger, aggregate_dataset

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
    
    logger = tb.SummaryWriter(path.join(args.log_dir, 'imitation'))
    
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
            logger.add_scalar('train loss', loss.item(), global_step)
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
        logger.add_scalar('success rate', success_rate, e)
        logger.add_scalar('mean reward', mean_reward, e)
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

def imitate(args):
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
    
    logger = tb.SummaryWriter(path.join(args.log_dir, 'imitation'))
    
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
                logger.add_scalar('train loss', loss.item(), global_step)
                
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
            logger.add_scalar('success rate', success_rate, e+1)
            logger.add_scalar('mean reward', mean_reward, e+1)
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

if __name__ == '__main__':
    # %load_ext tensorboard
    # %reload_ext tensorboard
    # %tensorboard --logdir="./logdir" --host=127.0.0.1 --host XXXX
    args = get_args()
    # train_model(args)
    final_positions, success_history, frames, reward_history, args = imitate(args)