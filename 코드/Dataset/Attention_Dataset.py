import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Attention_Dataset(Dataset):
    def __init__(self, model, table_1, table_2, table_3, embedding_dim, window_size):
        model.eval()

        dongs_apartment_complexes_embedding_matrixes_with_window_size = [] # (전체 동 개수 * 199, 38, window_size, 6)
        dongs_apartment_complexes_embedding_matrixes_with_window_size_num = [] # 단지 개수 # (전체 동 개수 * 199, 1)
        dongs_apartment_complexes_embedding_matrixes_with_window_size_index = [] # y 값이 있는 단지 index # (전체 동 개수 * 199, ?)
        dongs_apartment_complexes_prices_with_window_size = [] # (전체 동 개수 * 199, 38, 1)

        max_apartment_complexes = 38 # 최대 단지 개수

        table_1['dong'] = table_1['location'].apply(lambda x: x.split(' ')[2])
        dongs = table_1['dong'].unique()

        # 동 마다
        for dong in dongs: 
            # dong_apartment_complexes_embedding_matrixes(동 안의 단지마다 임베팅 matrix 구한 뒤 리스트 형식으로 모으기) 완성 # (동 안의 단지 개수, 204, 6)
            dong_apartment_complexes_values = table_1[table_1['dong'] == dong][[cols for cols in table_1.columns if cols not in ['aid','location','name','dong']]].values # 하나의 동 안의 아파트 단지 값들 # (동 안의 단지 개수, 10)
            economy_values = table_2[['call_rate','m2']].values # 경제 지표 값들 (204, 2)
            economy_tensor = torch.FloatTensor(economy_values) # 경제 지표 텐서 변환

            encoder_input_tensors = torch.zeros(dong_apartment_complexes_values.shape[0], 204, 12) # 인코더 입력 텐서들 초기화(인코더 입력 텐서 여러개) # (동 안의 단지 개수, 204(시점), 12)
            for i, dong_apartment_complex_values in enumerate(dong_apartment_complexes_values):
                dong_apartment_complex_tensor = torch.FloatTensor(dong_apartment_complex_values).repeat(204,1) 
                encoder_input_tensor = torch.cat((dong_apartment_complex_tensor, economy_tensor), dim=1)
                encoder_input_tensors[i] = encoder_input_tensor

            with torch.no_grad():
                dong_apartment_complexes_embedding_matrixes = torch.zeros(encoder_input_tensors.shape[0], 204, embedding_dim) # (동 안의 단지 개수, 204, 6)
                for i in range(encoder_input_tensors.shape[0]): # 동 안의 단지 (204, 6)
                    apartment_complex_embedding_matrix = torch.zeros(204,embedding_dim) # (204, 6)
                    for j in range(204): # 시점
                        apartment_complex_embedding_vector = model.encoder(encoder_input_tensors[i][j].unsqueeze(0)).squeeze() # (6, )
                        apartment_complex_embedding_matrix[j] = apartment_complex_embedding_vector
                    dong_apartment_complexes_embedding_matrixes[i] = apartment_complex_embedding_matrix


            # dong_apartment_complexes_prices(동 안의 단지마다 가격 구한 뒤 리스트 형식으로 모으기) 완성 # (동 안의 단지 개수, 204, 1)
            dong_apartment_complexes_aids = table_1[table_1['dong'] == dong]['aid'].values # (동 안의 단지 개수, )
            dong_apartment_complexes_prices = torch.zeros(dong_apartment_complexes_aids.shape[0],204,1) # (동 안의 단지 개수, 204, 1)
            for i, dong_apartment_complex_aid in zip(range(dong_apartment_complexes_aids.shape[0]), dong_apartment_complexes_aids): # 동 안의 단지 개수, 동 안의 단지들의 aids
                dong_apartment_complexes_prices[i] = torch.from_numpy(pd.DataFrame({'did': range(0, 204)}).merge(table_3[table_3['aid'] == dong_apartment_complex_aid][['did','price']], on='did', how='outer').fillna(0).set_index('did').values) # (204, 1)


            # dong_apartment_complexes_embedding_matrixes와 dong_apartment_complexes_prices window_size로 나누기
            for i in range(204-window_size): # window_size 고려한 시점(0~199)
                dong_apartment_complexes_embedding_matrixes_with_window_size = torch.zeros(max_apartment_complexes, window_size, embedding_dim) # (38, window_size, 6)
                dong_apartment_complexes_prices_with_window_size = torch.zeros(max_apartment_complexes, 1) # (38, 1)
                for j in range(dong_apartment_complexes_embedding_matrixes.shape[0]): # 동 안의 단지 개수
                    dong_apartment_complexes_embedding_matrixes_with_window_size[j] = dong_apartment_complexes_embedding_matrixes[j][i:i+window_size,:] # (window_size, 6)
                    dong_apartment_complexes_prices_with_window_size[j] = dong_apartment_complexes_prices[j][i+window_size,:] # (1, )
                dongs_apartment_complexes_embedding_matrixes_with_window_size_num.append(dong_apartment_complexes_embedding_matrixes.shape[0]) # 자연수
                dongs_apartment_complexes_embedding_matrixes_with_window_size_index.append(torch.nonzero(dong_apartment_complexes_prices_with_window_size, as_tuple=False)[:, 0]) # (1, )
                dongs_apartment_complexes_embedding_matrixes_with_window_size.append(dong_apartment_complexes_embedding_matrixes_with_window_size) # (38, window_size, 6)
                dongs_apartment_complexes_prices_with_window_size.append(dong_apartment_complexes_prices_with_window_size) # (38, 1)

        self.dongs_apartment_complexes_embedding_matrixes_with_window_size = dongs_apartment_complexes_embedding_matrixes_with_window_size
        self.dongs_apartment_complexes_embedding_matrixes_with_window_size_num = dongs_apartment_complexes_embedding_matrixes_with_window_size_num
        self.dongs_apartment_complexes_embedding_matrixes_with_window_size_index = dongs_apartment_complexes_embedding_matrixes_with_window_size_index
        self.dongs_apartment_complexes_prices_with_window_size = dongs_apartment_complexes_prices_with_window_size

    def __getitem__(self, i):
        # 임베딩(x), 단지 개수, y값 있는 단지 인덱스, 가격(y)
        return self.dongs_apartment_complexes_embedding_matrixes_with_window_size[i], self.dongs_apartment_complexes_embedding_matrixes_with_window_size_num[i], self.dongs_apartment_complexes_embedding_matrixes_with_window_size_index[i], self.dongs_apartment_complexes_prices_with_window_size[i]
    
    def __len__(self):
        return len(self.dongs_apartment_complexes_embedding_matrixes_with_window_size)