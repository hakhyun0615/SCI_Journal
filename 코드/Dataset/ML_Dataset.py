import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ML_Dataset(Dataset):
    def __init__(self, model, table_1, table_2, table_3, embedding_dim, window_size, DEVICE):
        # 동 이름 바꾸기
        old_and_new_dongs = {'용산동5가':'한강로동','한강로2가':'한강로동','창동':'창제동','돈암동':'정릉동','거여동':'위례동','문정동':'위례동','장지동':'위례동','문배동':'원효로동','산천동':'원효로동','신창동':'원효로동','원효로1가':'원효로동','화곡동':'우장산동','내발산동':'우장산동','영등포동8가':'영등포동','양평동3가':'양평동','안암동1가':'안암동','염리동':'아현동','성수동2가':'성수2가제2동','성수동1가':'성수1가제1동','중동':'성산동','노고산동':'서교동','신정동':'서강동','창전동':'서강동','삼선동4가':'삼선동','보문동3가':'보문동','동소문동7가':'동선동','당산동4가':'당산제2동','당산동5가':'당산제2동','당산동':'당산제2동','당산동3가':'당산제1동','당산동1가':'당산제1동','당산동2가':'당산제1동','본동':'노량진동','신수동':'노고산동','대흥동':'노고산동','금호동4가':'금호동','금호동2가':'금호동','충무로4가':'광희동','방화동':'공항동','도화동':'공덕동','신공덕동':'공덕동','일원동':'개포동'}
        def change_dongs(location):
            parts = location.split(' ')
            if parts[2] in old_and_new_dongs:
                parts[2] = old_and_new_dongs[parts[2]]
            return ' '.join(parts)
        table_1['location'] = table_1['location'].apply(change_dongs)

        # 정규화
        table_1[[cols for cols in table_1.columns if cols not in ['aid','location','name']]] = StandardScaler().fit_transform(table_1[[cols for cols in table_1.columns if cols not in ['aid','location','name']]])
        table_2[['call_rate','m2']] = StandardScaler().fit_transform(table_2[['call_rate','m2']])
        table_3['price'] = table_3['price'] * 0.0001

        model.eval()
        model.to(DEVICE)

        apartment_complexes_embedding_matrix_with_window_size = [] # (55135, window_size, embedding_dim)
        apartment_complexes_price_with_window_size = [] # (55135, 1)

        apartment_complexes_location = table_1['location']
        apartment_complexes_name = table_1['name']
        for apartment_complex_location, apartment_complex_name in zip(apartment_complexes_location, apartment_complexes_name): # 단지별로(702)
            apartment_complex_values = table_1[(table_1['name'] == apartment_complex_name) * (table_1['location'] == apartment_complex_location)][[cols for cols in table_1.columns if cols not in ['aid','location','name']]].values
            apartment_complex_tensor = torch.FloatTensor(apartment_complex_values).to(DEVICE).repeat(204, 1)
            economy_values = table_2[['call_rate','m2']].values
            economy_tensor = torch.FloatTensor(economy_values).to(DEVICE)
            encoder_input_tensor = torch.cat((apartment_complex_tensor, economy_tensor), dim=1) # 2006/01~2022/12까지(204) 12개의 features를 가지는 encoder_input_tensor 생성 # (204, 12)

            apartment_complex_embedding_matrix = np.zeros((encoder_input_tensor.shape[0], embedding_dim)) # (204, 1024)
            with torch.no_grad():
                for i in range(encoder_input_tensor.shape[0]): # 2006/01~2022/12까지 기간별로(204)
                    apartment_complex_embedding_vector = model.encoder(encoder_input_tensor[i].unsqueeze(0)).squeeze() # 12 features -> 1024 embedding_dim
                    if apartment_complex_embedding_vector.is_cuda:
                        apartment_complex_embedding_vector = apartment_complex_embedding_vector.cpu()
                    apartment_complex_embedding_matrix[i] = apartment_complex_embedding_vector.numpy()
            apartment_complex_embedding_matrix_tensor = torch.FloatTensor(apartment_complex_embedding_matrix).to(DEVICE) # (204, 1024)

            apartment_complex_aid = table_1[(table_1['name'] == apartment_complex_name) * (table_1['location'] == apartment_complex_location)]['aid'].squeeze()
            price_values = pd.DataFrame({'did': range(0, 204)}).merge(table_3[table_3['aid'] == apartment_complex_aid][['did','price']], on='did', how='outer').fillna(0).set_index('did').values
            price_tensor = torch.FloatTensor(price_values).to(DEVICE) # (204, 1)

            for i in range(apartment_complex_embedding_matrix_tensor.shape[0]-window_size):
                if price_tensor[i+window_size, :] != 0: # 가격이 있는 것만 취급
                    for window in range(window_size):
                        apartment_complex_embedding_matrix_concat_tensor = torch.zeros(1, embedding_dim * window_size)
                        apartment_complex_embedding_matrix_concat_tensor[:, window*embedding_dim:(window+1)*embedding_dim] = apartment_complex_embedding_matrix_tensor[i+window:i+window+1, :]
                    apartment_complexes_embedding_matrix_with_window_size.append(apartment_complex_embedding_matrix_concat_tensor) # (1, 10240)
                    apartment_complexes_price_with_window_size.append(price_tensor[i+window_size, :]) # (1, )

        self.apartment_complexes_embedding_matrix_with_window_size = apartment_complexes_embedding_matrix_with_window_size # (55135, 10240)
        self.apartment_complexes_price_with_window_size = apartment_complexes_price_with_window_size # (55135, 1)

    def __getitem__(self, i):
        return self.apartment_complexes_embedding_matrix_with_window_size[i], self.apartment_complexes_price_with_window_size[i]
    
    def __len__(self):
        return len(self.apartment_complexes_embedding_matrix_with_window_size)