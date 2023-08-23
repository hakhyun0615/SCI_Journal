import torch
from torch.utils.data import Dataset

class TransactionDataset(Dataset):
    def __init__(self, data, sequence_length=5):
        X, Y = [], []
        unique_complexes = list(data['단지명'].unique())
        for complex in unique_complexes:
            filtered_data_values = data[data['단지명']==complex]['평단가'].values
            for i in range(len(filtered_data_values) - sequence_length):
                X.append(filtered_data_values[i:i+sequence_length])
                Y.append(filtered_data_values[i+sequence_length])
        self.x = torch.FloatTensor(X)
        self.y = torch.FloatTensor(Y)
        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len