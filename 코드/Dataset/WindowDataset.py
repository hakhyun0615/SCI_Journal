import torch
from torch.utils.data import Dataset

# data : 시점, 부동산, 경제
# x1 : 경제 데이터, x2 : 부동산 데이터
# y1 : 경제 라벨링, x2 : 부동산 라벨링
class WindowDataset(Dataset):
      def __init__(self, data, step = 5):
            X1,X2,Y1,Y2 = [],[],[],[]
            data1 = data[['시점','경제']]
            data2 = data[['부동산']]
            for i in range(len(data)-step-1):
                X1.append(data1.iloc[i:i+step].values)
                Y1.append(data1.iloc[i+step].values)
                X2.append(data2.iloc[i:i+step].values)
                Y2.append(data2.iloc[i+step].values)

            self.x1 = torch.FloatTensor(X1)
            self.x2 = torch.FloatTensor(X2)
            self.y1 = torch.FloatTensor(Y1)
            self.y2 = torch.FloatTensor(Y2)
            
            self.len = len(X1)

      def __getitem__(self, i):
            return self.x1[i], self.x2[i], self.y1[i], self.y2[i]

      def __len__(self):
            return self.len