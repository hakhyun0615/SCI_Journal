import torch
from torch.utils.data import Dataset

class EconomyDataset(Dataset):
      def __init__(self, data, step = 5):
            data = df['국고채금리']
            X,Y = [],[]
            for i in range(len(data)-step-1):
                  X.append(data[i:i+step])
                  Y.append(data[i+step])

            self.x = torch.FloatTensor(X)
            self.y = torch.FloatTensor(Y)

            self.len = len(X)

      def __getitem__(self, i):
            return self.x[i], self.y[i]

      def __len__(self):
            return self.len