import torch
from torch import nn
import matplotlib.pyplot as plt

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE,self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, y, y_hat):
        return torch.sqrt(self.mse(y, y_hat))

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, y_hat, y):
        return self.mse(y_hat, y)

class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
    def forward(self, y_hat, y):
        return torch.mean(torch.abs(y_hat - y))

class MAPE(nn.Module):
    def __init__(self):
        super(MAPE, self).__init__()
    def forward(self, y_hat, y):
        return torch.mean(torch.abs((y_hat - y) / (y + 1e-8))) * 100

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
