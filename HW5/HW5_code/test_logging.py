import torch
import torch.utils.tensorboard as tb
from torch.utils.tensorboard import SummaryWriter
from os import path

def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    # This is a strongly simplified training loop
    for epoch in range(10):
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            train_logger.add_scalar('train loss',dummy_train_loss,0)
        # train_logger.add_scalar('train accuracy',dummy_train_accuracy,0)
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
        # valid_logger.add_scalar('valid accuracy',dummy_validation_accuracy,0)
        
# input following command in terminal
# %load_ext tensorboard
# %reload_ext tensorboard
# %tensorboard --logdir {ROOT_LOG_DIR}
        
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ROOT_LOG_DIR = './logdir'
    train_logger = tb.SummaryWriter(path.join('./logdir', 'train'))
    valid_logger = tb.SummaryWriter(path.join('./logdir', 'test'))
    test_logging(train_logger, valid_logger)