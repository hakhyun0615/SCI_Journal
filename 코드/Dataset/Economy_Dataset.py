import torch
from torch.utils.data import Dataset

class EconomyDataset(Dataset):
      def __init__(self, data, sequence_length=5):
            X,Y = [],[]
            for i in range(len(data)-sequence_length-1):
                  X.append(data[i:i+sequence_length])
                  Y.append(data[i+sequence_length])

            self.x = torch.FloatTensor(X)
            self.y = torch.FloatTensor(Y)

            self.len = len(X)

      def __getitem__(self, i):
            return self.x[i], self.y[i]

      def __len__(self):
            return self.len
      
