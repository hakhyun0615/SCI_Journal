import torch
from torch.utils.data import Dataset

class RNN_Economy_Dataset(Dataset):
    def __init__(self, data, sequence_length=5):
        nation_economy_x, nation_economy_y, call_economy_x, call_economy_y = [], [], [], []
        for i in range(len(data)-sequence_length-1):
            nation_economy_x.append(data['국고채금리'][i:i+sequence_length].to_list())
            nation_economy_y.append(data['국고채금리'][i+sequence_length:i+sequence_length+1].to_list())
            call_economy_x.append(data['콜금리'][i:i+sequence_length].to_list())
            call_economy_y.append(data['콜금리'][i+sequence_length:i+sequence_length+1].to_list())

        self.nation_economy_x = torch.FloatTensor(nation_economy_x)
        self.nation_economy_y = torch.FloatTensor(nation_economy_y)
        self.call_economy_x = torch.FloatTensor(call_economy_x)
        self.call_economy_y = torch.FloatTensor(call_economy_y)
        self.len = len(nation_economy_x)

    def __getitem__(self, i):
        return self.nation_economy_x[i], self.nation_economy_y[i], self.call_economy_x[i], self.call_economy_y[i]

    def __len__(self):
        return self.len
      
