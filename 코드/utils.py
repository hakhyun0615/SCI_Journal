import torch
from torch import nn
import matplotlib.pyplot as plt

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE,self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, y, y_hat):
        return torch.sqrt(self.mse(y, y_hat))

def plot_train_val_losses(train_losses, val_losses, save_path):
    print(f'Min Train Loss: {min(train_losses)}')
    print(f'Min Validation Loss: {min(val_losses)}')
    plt.plot(train_losses[1:], label='Training Loss')
    plt.plot(val_losses[1:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(save_path)