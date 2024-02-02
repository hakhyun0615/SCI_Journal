import torch
from torch import nn
import matplotlib.pyplot as plt

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE,self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, y_true, y_pred):
        return torch.sqrt(self.mse(y_true, y_pred))

def rmse(y_pred, y_true): 
    mse = torch.mean((y_true - y_pred) ** 2)
    return torch.sqrt(mse)

def mse(y_pred, y_true): 
    return torch.mean((y_true - y_pred) ** 2)

def mae(y_pred, y_true):
    return torch.mean(torch.abs(y_true - y_pred))

def save_train_val_losses(train_losses, val_losses, save_path):
    print(f'Min Train Loss: {min(train_losses)}')
    print(f'Min Val Loss: {min(val_losses)}')

    plt.clf()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path+'.jpg')
    
    with open(f'{save_path}_train_losses.txt', 'w') as f:
        for item in train_losses:
            f.write("%s\n" % item)

    with open(f'{save_path}_val_losses.txt', 'w') as f:
        for item in val_losses:
            f.write("%s\n" % item)
