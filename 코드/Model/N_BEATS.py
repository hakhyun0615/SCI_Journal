import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, horizon, n_neurons, n_layers):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        self.fc_layers = self._create_fc_layers()
        self.theta_layer = nn.Linear(n_neurons, theta_size)

    def _create_fc_layers(self):
        layers = []
        for i in range(self.n_layers):
            if i == 0:
                layers.append(nn.Linear(self.input_size, self.n_neurons))
            else:
                layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast

class NBeats(nn.Module):
    def __init__(self, input_size, horizon, n_stacks, n_blocks, n_neurons, n_layers, theta_size):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.n_stacks = n_stacks
        self.n_blocks = n_blocks

        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, theta_size, horizon, n_neurons, n_layers)
            for _ in range(n_stacks * n_blocks)
        ])

    def forward(self, x):
        residuals = x.clone()
        forecast = torch.zeros(x.size(0), self.horizon, device=x.device)

        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast

        return forecast