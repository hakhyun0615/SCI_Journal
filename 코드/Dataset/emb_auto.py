import torch
from torch.utils.data import Dataset

# 필요하다면 파라미터 개수를 늘려
class AutoDataset(Dataset):
    def __init__(self, data):

        data_len = len(data)
        X, Y =[],[]
        for j in range(data_len):
          X.append(data[0].values[j])

        for j in range(data_len):
          Y.append(data[1].values[j])

        self.x = torch.FloatTensor(X)
        self.y = torch.FloatTensor(Y)

        self.len = len(X)

    # return 값들
    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len