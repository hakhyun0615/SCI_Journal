import torch
from torch import nn
import matplotlib.pyplot as plt

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE,self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, y, y_hat):
        return torch.sqrt(self.mse(y, y_hat))

def save_train_val_losses(train_losses, val_losses, save_path):
    print(f'Min Train Loss: {min(train_losses)}')
    print(f'Min Validation Loss: {min(val_losses)}')
    
    with open(f'{save_path}_train_losses.txt', 'w') as f:
        for item in train_losses:
            f.write("%s\n" % item)

    with open(f'{save_path}_val_losses.txt', 'w') as f:
        for item in val_losses:
            f.write("%s\n" % item)
