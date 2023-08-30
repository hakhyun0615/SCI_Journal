import torch
from torch.utils.data import Dataset

class RNN_Economy_Dataset(Dataset):
    def __init__(self, data, sequence_length=5):
        economy_x, economy_y = [], [],
        for idx in range(len(data)-sequence_length):
            economy_x.append(data[idx:idx+sequence_length].to_list())
            economy_y.append(data[idx+sequence_length:idx+sequence_length+1].to_list())

        self.economy_x = torch.FloatTensor(economy_x)
        self.economy_y = torch.FloatTensor(economy_y)
        self.len = len(economy_x)

    def __getitem__(self, i):
        return self.economy_x[i], self.economy_y[i]

    def __len__(self):
        return self.len
      
