import torch
import torch.utils.tensorboard as tb
from os import path

def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """
    global_step = 0
    size_train = 20
    size_valid = 10
    for epoch in range(10):
        torch.manual_seed(epoch)
        train_acc = []
        valid_acc = []
        for iteration in range(size_train):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            train_logger.add_scalar('train loss',
                                    dummy_train_loss,
                                    global_step)
            train_acc.append(torch.mean(dummy_train_accuracy))
            global_step += 1
        
        train_acc_iter = torch.mean(torch.FloatTensor(train_acc))
        train_logger.add_scalar('train accuracy',
                                train_acc_iter,
                                epoch)
        
        torch.manual_seed(epoch)
        for iteration in range(size_valid):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            valid_acc.append(torch.mean(dummy_validation_accuracy))
            
        valid_acc_iter = torch.mean(torch.FloatTensor(valid_acc))
        valid_logger.add_scalar('valid accuracy',
                                valid_acc_iter,
                                epoch)
        
if __name__ == '__main__':
    # %load_ext tensorboard
    # %reload_ext tensorboard
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ROOT_LOG_DIR = './logdir'
    # %tensorboard --logdir {ROOT_LOG_DIR} --host=127.0.0.1
    train_logger = tb.SummaryWriter(path.join('./logdir', 'train'))
    valid_logger = tb.SummaryWriter(path.join('./logdir', 'test'))
    test_logging(train_logger, valid_logger)