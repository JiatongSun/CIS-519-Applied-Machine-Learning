import torch
import matplotlib.pyplot as plt
from args import get_args
from load_expert_data import load_initial_data, load_dataset
from test_model import test_model
from dagger import execute_dagger, aggregate_dataset

def train_model(args):    
    model = args.model
    optimizer = args.optimizer
    criterion = args.criterion
    dataset_sizes = args.dataset_sizes
    dataloader = args.dataloader
    logger = args.logger
    
    global global_step
    global loss_list
       
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
        loss_list.append(loss.item())
        logger.add_scalar('training loss', loss.item(), global_step)
    epoch_loss = running_loss / dataset_sizes
    epoch_acc = running_corrects.double() / dataset_sizes
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    return model

def imitate(args):
    # initialization
    epoch = args.num_epochs
    train_obs, train_act = load_initial_data(args)
    batch = args.batch_size
    transform = args.transform
    
    args.dataset_sizes = len(train_obs)
    args.dataloader = load_dataset(args,train_obs,train_act,batch, transform)
    num_valid_episode = args.num_valid_episode
    
    logger = args.logger
        
    final_positions = []
    success_history = []
    frames = []
    reward_history = []
        
    if args.do_dagger: epoch += args.max_dagger_iterations

    for e in range(epoch):
        # train
        print('epoch: {}'.format(e+1))
        args.model = train_model(args)
        
        # dagger
        if args.do_dagger and e>=args.num_epochs:
            imit_obs, exp_act = execute_dagger(args)
            train_obs, train_act = aggregate_dataset(train_obs, train_act, 
                                                     imit_obs, exp_act)
            args.dataset_sizes = len(train_obs)
            args.dataloader = load_dataset(args,train_obs,train_act,batch, transform)
        
        # valid
        total_reward = 0
        total_success = 0
        for i_episode in range(num_valid_episode):
            final_position, success, frames_ep, episode_reward \
                = test_model(args)
            total_success += success
            total_reward += episode_reward
            final_positions.append(final_position)
            # success_history.append(success)
            # reward_history.append(episode_reward)
        frames.append(frames_ep)
        success_rate = total_success / num_valid_episode
        mean_reward = total_reward / num_valid_episode
        
        success_history.append(success_rate)
        reward_history.append(mean_reward)
        
        logger.add_scalar('success rate', success_rate, e+1)
        logger.add_scalar('mean reward', mean_reward, e+1)
        print('success rate: {} mean reward: {}'.format(success_rate, 
                                                        mean_reward))
    
    return final_positions, success_history, frames, reward_history, args

if __name__ == '__main__':
    # %load_ext tensorboard
    # %reload_ext tensorboard
    # %tensorboard --logdir="./logdir" --host=127.0.0.1 --host XXXX
    
    global_step = 0
    loss_list = []
    
    args = get_args()
    positions, successes, frames, reward_history, args = imitate(args)
    
    plt.plot(successes)
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.title('Success Rate')
    plt.show()
    plt.close()
    
    plt.plot(reward_history)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward')
    plt.show()
    plt.close()
    
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')
    plt.show()
    plt.close()