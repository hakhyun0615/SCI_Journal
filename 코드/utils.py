import torch
from torch import nn
import matplotlib.pyplot as plt

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE,self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, y_true, y_pred):
        return torch.sqrt(self.mse(y_true, y_pred))

def rmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    return torch.sqrt(mse)

def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def mape(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

def save_train_test_losses(train_losses, test_losses, save_path):
    print(f'Min Train Loss: {min(train_losses)}')
    print(f'Min Test Loss: {min(test_losses)}')

    plt.clf()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path+'.jpg')
    
    with open(f'{save_path}_train_losses.txt', 'w') as f:
        for item in train_losses:
            f.write("%s\n" % item)

    with open(f'{save_path}_test_losses.txt', 'w') as f:
        for item in test_losses:
            f.write("%s\n" % item)
