import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.unsqueeze(1))
        x = x.squeeze(1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual

class DLinear(nn.Module):
    def __init__(self, input_size, horizon, kernel_size=25):
        super().__init__()
        self.decomposition = SeriesDecomp(kernel_size)
        self.linear_trend = nn.Linear(input_size, horizon)
        self.linear_seasonal = nn.Linear(input_size, horizon)

    def forward(self, x):
        trend, seasonal = self.decomposition(x)
        trend_output = self.linear_trend(trend)
        seasonal_output = self.linear_seasonal(seasonal)
        x = trend_output + seasonal_output
        return x