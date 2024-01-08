import torch
from torch import nn
import matplotlib.pyplot as plt

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE,self).__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-7

    def forward(self, y, y_hat):
        return torch.sqrt(self.mse(y, y_hat) + self.eps)

def plot_train_val_losses(train_losses, val_losses):
    print(f'Min Validation Loss: {min(val_losses)}')
    plt.plot(train_losses[1:], label='Training Loss')
    plt.plot(val_losses[1:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

# val loss가 연속적으로 오를 때
def early_stop_1(val_losses, consecutive_val_loss_increases, max_consecutive_val_loss_increases):
    if len(val_losses) > 1 and val_losses[-1] >= val_losses[-2]:
        consecutive_val_loss_increases += 1
        if consecutive_val_loss_increases >= max_consecutive_val_loss_increases:
            return True, consecutive_val_loss_increases
        else:
            return False, consecutive_val_loss_increases
    else:
        consecutive_val_loss_increases = 0
        return False, consecutive_val_loss_increases

# val loss가 최저 loss보다 연속적으로 클 때
def early_stop_2(avg_val_loss, best_val_loss, consecutive_val_loss_increases, max_consecutive_val_loss_increases):
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        consecutive_val_loss_increases = 0
    else:
        consecutive_val_loss_increases += 1
    if consecutive_val_loss_increases >= max_consecutive_val_loss_increases:
        return True, best_val_loss, consecutive_val_loss_increases
    else:
        return False, best_val_loss, consecutive_val_loss_increases