import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class Dong_Dataset(Dataset):
    def __init__(self, model, table_1, table_2, table_3, embedding_dim, window_size, DEVICE):
        # 데이터프레임 복사본 생성
        table_1_copy = table_1.copy()
        table_2_copy = table_2.copy()
        table_3_copy = table_3.copy()

        # 정규화
        scaler = StandardScaler()
        table_1_copy[[cols for cols in table_1_copy.columns if cols not in ['aid','location','name']]] = scaler.fit_transform(table_1_copy[[cols for cols in table_1_copy.columns if cols not in ['aid','location','name']]])
        scaler.fit(table_2_copy[[cols for cols in table_2_copy.columns if cols not in ['did','year','month']]][:135])
        table_2_copy[[cols for cols in table_2_copy.columns if cols not in ['did','year','month']]] = scaler.transform(table_2_copy[[cols for cols in table_2_copy.columns if cols not in ['did','year','month']]])
        table_3_copy['price'] = table_3_copy['price'] * 0.0001 # 억 단위

        # 동 이름 바꾸기
        old_and_new_dongs = {'용산동5가':'한강로동','한강로2가':'한강로동','창동':'창제동','돈암동':'정릉동','거여동':'위례동','문정동':'위례동','장지동':'위례동','문배동':'원효로동','산천동':'원효로동','신창동':'원효로동','원효로1가':'원효로동','화곡동':'우장산동','내발산동':'우장산동','영등포동8가':'영등포동','양평동3가':'양평동','안암동1가':'안암동','염리동':'아현동','성수동2가':'성수2가제2동','성수동1가':'성수1가제1동','중동':'성산동','노고산동':'서교동','신정동':'서강동','창전동':'서강동','삼선동4가':'삼선동','보문동3가':'보문동','동소문동7가':'동선동','당산동4가':'당산제2동','당산동5가':'당산제2동','당산동':'당산제2동','당산동3가':'당산제1동','당산동1가':'당산제1동','당산동2가':'당산제1동','본동':'노량진동','신수동':'노고산동','대흥동':'노고산동','금호동4가':'금호동','금호동2가':'금호동','충무로4가':'광희동','방화동':'공항동','도화동':'공덕동','신공덕동':'공덕동','일원동':'개포동'}
        def change_dongs(location):
            parts = location.split(' ')
            if parts[2] in old_and_new_dongs:
                parts[2] = old_and_new_dongs[parts[2]]
            return ' '.join(parts)
        table_1_copy['location'] = table_1_copy['location'].apply(change_dongs)
 
        # 동 종류
        table_1_copy['dong'] = table_1_copy['location'].apply(lambda x: x.split(' ')[2])
        dongs = table_1_copy['dong'].unique()

        # (동별) 최대 단지 개수
        # TRAIN: 38 
        # TEST: 24
        max_apartment_complexes = max(table_1_copy.groupby('dong')['name'].count())

        # (전체 동 개수 * 204-window_size, 최대 단지 개수, window_size, embedding_dim) 
        # TRAIN: (22698, 38, 10, 1024)
        # TEST: (1120, 24, 10, 1024)
        dongs_apartment_complexes_embedding_matrixes_with_window_size = [] 
        # 단지 개수 
        # (전체 동 개수 * 204-window_size, 1)
        dongs_apartment_complexes_embedding_matrixes_with_window_size_num = [] 
        # y 값이 있는 단지 index 
        # (전체 동 개수 * 204-window_size, ?)
        dongs_apartment_complexes_embedding_matrixes_with_window_size_index = [] 
        # (전체 동 개수 * 204-window_size, 최대 단지 개수, 1)
        # TRAIN: (22698, 38, 1)
        # TEST: (1120, 24, 1)
        dongs_apartment_complexes_prices_with_window_size = [] 

        if model != 'None': # 임베딩 벡터를 사용할 때
            model.eval()
            model.to(DEVICE)

        # 동 마다
        for dong in dongs: 
            # dong_apartment_complexes_embedding_matrixes(동 안의 단지마다 임베팅 matrix 구한 뒤 리스트 형식으로 모으기) 완성 # (동 안의 단지 개수, 204, 6)
            dong_apartment_complexes_values = table_1_copy[table_1_copy['dong'] == dong][[cols for cols in table_1_copy.columns if cols not in ['aid','location','name','dong']]].values # 하나의 동 안의 아파트 단지 값들 # (동 안의 단지 개수, 10)
            economy_values = table_2_copy[['call_rate','m2']].values # 경제 지표 값들 # (204/20, 2)
            economy_tensor = torch.FloatTensor(economy_values).to(DEVICE) # 경제 지표 텐서 변환

            encoder_input_tensors = torch.zeros(dong_apartment_complexes_values.shape[0], len(table_2_copy), 12).to(DEVICE) # 인코더 입력 텐서들 초기화(인코더 입력 텐서 여러개) # (동 안의 단지 개수, 204(시점), 12)
            for i, dong_apartment_complex_values in enumerate(dong_apartment_complexes_values):
                dong_apartment_complex_tensor = torch.FloatTensor(dong_apartment_complex_values).to(DEVICE).repeat(len(table_2_copy),1) 
                encoder_input_tensor = torch.cat((dong_apartment_complex_tensor, economy_tensor), dim=1)
                encoder_input_tensors[i] = encoder_input_tensor

            if embedding_dim != 'None': # 임베딩 벡터를 사용할 때
                with torch.no_grad():
                    dong_apartment_complexes_embedding_matrixes = torch.zeros(encoder_input_tensors.shape[0], len(table_2_copy), embedding_dim) # (동 안의 단지 개수, 204/20, 1024)
                    for i in range(encoder_input_tensors.shape[0]): # 동 안의 단지 (204/20, 1024)
                        apartment_complex_embedding_matrix = torch.zeros(len(table_2_copy), embedding_dim) # (204/20, 1024)
                        for j in range(len(table_2_copy)): # 시점
                            apartment_complex_embedding_vector = model.encoder(encoder_input_tensors[i][j].unsqueeze(0)).squeeze() # (1024, )
                            apartment_complex_embedding_matrix[j] = apartment_complex_embedding_vector
                        dong_apartment_complexes_embedding_matrixes[i] = apartment_complex_embedding_matrix

            # dong_apartment_complexes_prices(동 안의 단지마다 가격 구한 뒤 리스트 형식으로 모으기) 완성 # (동 안의 단지 개수, 204/20, 1)
            dong_apartment_complexes_aids = table_1_copy[table_1_copy['dong'] == dong]['aid'].values # (동 안의 단지 개수, )
            dong_apartment_complexes_prices = torch.zeros(dong_apartment_complexes_aids.shape[0], len(table_2_copy), 1).to(DEVICE) # (동 안의 단지 개수, 204/20, 1)
            for i, dong_apartment_complex_aid in zip(range(dong_apartment_complexes_aids.shape[0]), dong_apartment_complexes_aids): # 동 안의 단지 개수, 동 안의 단지들의 aids
                dong_apartment_complexes_prices[i] = torch.from_numpy(pd.DataFrame({'did': range(0, len(table_2_copy))}).merge(table_3_copy[table_3_copy['aid'] == dong_apartment_complex_aid][['did','price']], on='did', how='outer').fillna(0).set_index('did').values) # (204/20, 1)

            if embedding_dim == 'None': # 임베딩 벡터가 없을 때
                dong_apartment_complexes_embedding_matrixes = encoder_input_tensors
            # dong_apartment_complexes_embedding_matrixes와 dong_apartment_complexes_prices window_size로 나누기
            for i in range(len(table_2_copy)-window_size): # window_size 고려한 시점(0~199/19)
                if embedding_dim == 'None': # 임베딩 벡터가 없을 때
                    dong_apartment_complexes_embedding_matrixes_with_window_size = torch.zeros(max_apartment_complexes, window_size, 12)
                else:
                    dong_apartment_complexes_embedding_matrixes_with_window_size = torch.zeros(max_apartment_complexes, window_size, embedding_dim) # (38/224, window_size, 1024)
                dong_apartment_complexes_prices_with_window_size = torch.zeros(max_apartment_complexes, 1).to(DEVICE) # (38/24, 1)
                for j in range(dong_apartment_complexes_embedding_matrixes.shape[0]): # 동 안의 단지 개수
                    dong_apartment_complexes_embedding_matrixes_with_window_size[j] = dong_apartment_complexes_embedding_matrixes[j][i:i+window_size,:] # (window_size, 1024)
                    dong_apartment_complexes_prices_with_window_size[j] = dong_apartment_complexes_prices[j][i+window_size,:] # (1, )
                dongs_apartment_complexes_embedding_matrixes_with_window_size_num.append(dong_apartment_complexes_embedding_matrixes.shape[0]) # 자연수
                dongs_apartment_complexes_embedding_matrixes_with_window_size_index.append(torch.nonzero(dong_apartment_complexes_prices_with_window_size, as_tuple=False)[:, 0]) # (1, )
                dongs_apartment_complexes_embedding_matrixes_with_window_size.append(dong_apartment_complexes_embedding_matrixes_with_window_size) # (38, window_size, 1024)
                dongs_apartment_complexes_prices_with_window_size.append(dong_apartment_complexes_prices_with_window_size) # (38/24, 1)

        self.dongs_apartment_complexes_embedding_matrixes_with_window_size = dongs_apartment_complexes_embedding_matrixes_with_window_size
        self.dongs_apartment_complexes_embedding_matrixes_with_window_size_num = dongs_apartment_complexes_embedding_matrixes_with_window_size_num
        self.dongs_apartment_complexes_embedding_matrixes_with_window_size_index = dongs_apartment_complexes_embedding_matrixes_with_window_size_index
        self.dongs_apartment_complexes_prices_with_window_size = dongs_apartment_complexes_prices_with_window_size

    def __getitem__(self, i):
        # 임베딩(x), 단지 개수, y값 있는 단지 인덱스, 가격(y)
        return self.dongs_apartment_complexes_embedding_matrixes_with_window_size[i], self.dongs_apartment_complexes_embedding_matrixes_with_window_size_num[i], self.dongs_apartment_complexes_embedding_matrixes_with_window_size_index[i], self.dongs_apartment_complexes_prices_with_window_size[i]
    
    def __len__(self):
        return len(self.dongs_apartment_complexes_embedding_matrixes_with_window_size)