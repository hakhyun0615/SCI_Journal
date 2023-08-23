import torch
from torch.utils.data import Dataset


class TransactionDataset(Dataset):
    def __init__(self, data, address, complex, sequence_length=5):
        filtered_data = data[(data['시군구'] == address) & (data['단지명'] == complex)]
        data_values = filtered_data['평단가'].values
        
        X, Y = [], []
        for i in range(len(data_values) - sequence_length):
            X.append(data_values[i:i+sequence_length])
            Y.append(data_values[i+sequence_length])
        
        self.x = torch.FloatTensor(X)
        self.y = torch.FloatTensor(Y)
        
        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len