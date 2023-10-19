import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import price_interpolate

class LSTM_Transaction_Train_Dataset(Dataset):
    def __init__(self, df, window_size=5):
        df['계약년월'] = pd.to_datetime(df['계약년월'].astype(str), format='%Y%m')
        interpolated_df = pd.DataFrame(df.groupby(['동', '단지']).apply(lambda group: price_interpolate(group,group['계약년월'].min(),group['계약년월'].max()))['제곱미터당 거래금액']).reset_index().rename(columns={'level_2':'계약년월'})

        dongs_x, dongs_y = [], []
        for dong in interpolated_df['동'].unique():
            for apartment_complex in interpolated_df[interpolated_df['동'] == dong]['단지'].unique():
                filtered_interpolated_df = interpolated_df[(interpolated_df['동'] == dong)*(interpolated_df['단지'] == apartment_complex)]
                filtered_interpolated_df_values = filtered_interpolated_df['제곱미터당 거래금액'].values
                filtered_interpolated_df_times = filtered_interpolated_df['계약년월'].apply(lambda x: float((x.year-pd.Timestamp('2006-01').year)*12+(x.month-pd.Timestamp('2006-01').month)+1)).values
                for idx in range(len(filtered_interpolated_df_values)-window_size):
                    dongs_x.append([filtered_interpolated_df_values[idx:idx+window_size],filtered_interpolated_df_times[idx:idx+window_size]])
                    dongs_y.append([filtered_interpolated_df_values[idx+window_size:idx+window_size+1],filtered_interpolated_df_times[idx+window_size:idx+window_size+1]])
        self.dongs_x = dongs_x
        self.dongs_y = dongs_y
        self.len = len(dongs_x)

    # 부동산_x, 부동산_시점_x, 부동산_y, 부동산_시점_y 
    def __getitem__(self, i):
        return torch.FloatTensor(self.dongs_x[i][0]), torch.FloatTensor(self.dongs_x[i][1]), torch.FloatTensor(self.dongs_y[i][0]), torch.FloatTensor(self.dongs_y[i][1])

    def __len__(self):
        return self.len