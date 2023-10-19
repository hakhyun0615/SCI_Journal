import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import price_interpolate

class LSTM_Transaction_Val_Dataset(Dataset):
    def __init__(self, df, window_size=5):
        df['계약년월'] = pd.to_datetime(df['계약년월'].astype(str), format='%Y%m')
        df_2022 = df[df['계약년월'].dt.year == 2022]
        df_else = df[df['계약년월'].dt.year != 2022]

        dongs_x, dongs_y = [], []
        for dong in df_2022['동'].unique():
            for apartment_complex in df_2022[df_2022['동'] == dong]['단지'].unique():
                # 2022년 이전 거래 window_size만큼 가져와 2022년 첫 거래 예측
                filtered_df_2022 = df_2022[(df_2022['동'] == dong)*(df_2022['단지'] == apartment_complex)]
                filtered_df_2022_values = filtered_df_2022['제곱미터당 거래금액'].values
                filtered_df_2022_times = filtered_df_2022['계약년월'].apply(lambda x: float((x.year-pd.Timestamp('2006-01').year)*12+(x.month-pd.Timestamp('2006-01').month)+1)).values

                filtered_df_else = df_else[(df_else['동'] == dong)*(df_else['단지'] == apartment_complex)]
                past_date_range = pd.date_range(start='2006-01-01', end=filtered_df_else.iloc[-1]['계약년월']-pd.DateOffset(months=window_size), freq='MS')
                interpolate_date_range = pd.date_range(start=filtered_df_else[filtered_df_else['계약년월'].isin(past_date_range)].iloc[-1]['계약년월'], end=filtered_df_else.iloc[-1]['계약년월'], freq='MS')
                
                double_filtered_df_else = price_interpolate(filtered_df_else[filtered_df_else['계약년월'].isin(interpolate_date_range)],interpolate_date_range[0],interpolate_date_range[-1]).reset_index().rename(columns={'index': '계약년월'})[-window_size:]
                double_filtered_df_else_values = double_filtered_df_else['제곱미터당 거래금액'].values
                double_filtered_df_times = double_filtered_df_else['계약년월'].apply(lambda x: float((x.year-pd.Timestamp('2006-01').year)*12+(x.month-pd.Timestamp('2006-01').month)+1)).values
                dongs_x.append([double_filtered_df_else_values,double_filtered_df_times])
                dongs_y.append([filtered_df_2022_values[0:1],filtered_df_2022_times[0:1]])
                
                # 2022년 나머지 거래 예측
                if len(filtered_df_2022) > 1:
                    filtered_df = df[(df['동'] == dong)*(df['단지'] == apartment_complex)]
                    for idx in range(len(filtered_df_2022)-1):
                        past_date_range = pd.date_range(start='2006-01-01', end=filtered_df_2022.iloc[idx]['계약년월']-pd.DateOffset(months=window_size), freq='MS')
                        interpolate_date_range = pd.date_range(start=filtered_df[filtered_df['계약년월'].isin(past_date_range)].iloc[-1]['계약년월'], end=filtered_df_2022.iloc[idx]['계약년월'], freq='MS')
                        double_filtered_df = price_interpolate(filtered_df[filtered_df['계약년월'].isin(interpolate_date_range)],interpolate_date_range[0],interpolate_date_range[-1]).reset_index().rename(columns={'index': '계약년월'})[-window_size:]
                        double_filtered_df_values = double_filtered_df['제곱미터당 거래금액'].values
                        double_filtered_df_times = double_filtered_df['계약년월'].apply(lambda x: float((x.year-pd.Timestamp('2006-01').year)*12+(x.month-pd.Timestamp('2006-01').month)+1)).values
                        dongs_x.append([double_filtered_df_values,double_filtered_df_times])
                        dongs_y.append([filtered_df_2022_values[idx+1:idx+2],filtered_df_2022_times[idx+1:idx+2]])
                else:
                    continue

        self.dongs_x = dongs_x
        self.dongs_y = dongs_y
        self.len = len(dongs_x)

    # 부동산_x, 부동산_시점_x, 부동산_y, 부동산_시점_y 
    def __getitem__(self, i):
        return torch.FloatTensor(self.dongs_x[i][0]), torch.FloatTensor(self.dongs_x[i][1]), torch.FloatTensor(self.dongs_y[i][0]), torch.FloatTensor(self.dongs_y[i][1])

    def __len__(self):
        return self.len