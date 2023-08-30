import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import price_per_pyeong_fill_0

class RNN_Transformer_Dataset(Dataset):
    def __init__(self, transaction_data, economy_data, sequence_length=5):
        all_dong_max_apartment_complex = transaction_data.drop_duplicates(subset=['시군구','단지명']).groupby(['시군구'])['단지명'].count().max()

        filled_data = price_per_pyeong_fill_0(transaction_data)
        filled_data = filled_data[['시군구', '단지명', '계약년월', '평단가']]        

        dongs_x, dongs_y = [], []
        for dong in filled_data['시군구'].unique():
            filtered_filled_data = filled_data[filled_data['시군구'] == dong]
            date_range = pd.date_range('20060101', '20221201', freq='MS')
            for idx in range(len(date_range)-sequence_length):
                current_range_x = date_range[idx:idx+sequence_length]
                current_range_y = date_range[idx+sequence_length:idx+sequence_length+1]
                current_range_filled_x = filtered_filled_data[filtered_filled_data['계약년월'].isin(current_range_x)]
                current_range_filled_y = filtered_filled_data[filtered_filled_data['계약년월'].isin(current_range_y)]
                grouped_current_range_filled_x = current_range_filled_x.groupby('단지명').agg({'평단가': list}).reset_index()['평단가'].to_list()
                grouped_current_range_filled_y = current_range_filled_y.groupby('단지명').agg({'평단가': list}).reset_index()['평단가'].to_list()
                if len(grouped_current_range_filled_x) < all_dong_max_apartment_complex:
                    for _ in range(all_dong_max_apartment_complex-len(grouped_current_range_filled_x)):
                        grouped_current_range_filled_x.append([0.0]*sequence_length)
                        grouped_current_range_filled_y.append([0.0])
                economy_x, economy_y = [], []
                economy_x.append(economy_data[idx:idx+sequence_length].to_list())
                economy_y.append(economy_data[idx+sequence_length:idx+sequence_length+1].to_list())
                grouped_current_range_filled_and_economy_x = []
                grouped_current_range_filled_and_economy_x.extend([grouped_current_range_filled_x, economy_x])
                grouped_current_range_filled_and_economy_y = []
                grouped_current_range_filled_and_economy_y.extend([grouped_current_range_filled_y, economy_y])
                dongs_x.append(grouped_current_range_filled_and_economy_x)
                dongs_y.append(grouped_current_range_filled_and_economy_y)

        self.dongs_x = dongs_x
        self.dongs_y = dongs_y
        self.len = len(dongs_x)

    # 부동산_x, 경제_x, 부동산_y, 경제_y 
    def __getitem__(self, i):
        return torch.FloatTensor(self.dongs_x[i][0]), torch.FloatTensor(self.dongs_x[i][1]), torch.FloatTensor(self.dongs_y[i][0]), torch.FloatTensor(self.dongs_y[i][1])

    def __len__(self):
        return self.len