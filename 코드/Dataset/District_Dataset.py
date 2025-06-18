import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import gc  # 메모리 정리용

class District_Dataset(Dataset):
    def __init__(self, model, table_1, table_2, table_3, embedding_dim, window_size, SUB, DEVICE):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.DEVICE = DEVICE
        
        # 메모리 효율을 위해 CPU에서 처리 후 필요할 때만 GPU로 이동
        self.device_cpu = torch.device('cpu')
        
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
 
        if SUB == True:
            # 동 이름 바꾸기
            old_and_new_dongs = {'용산동5가':'한강로동','한강로2가':'한강로동','창동':'창제동','돈암동':'정릉동','거여동':'위례동','문정동':'위례동','장지동':'위례동','문배동':'원효로동','산천동':'원효로동','신창동':'원효로동','원효로1가':'원효로동','화곡동':'우장산동','내발산동':'우장산동','영등포동8가':'영등포동','양평동3가':'양평동','안암동1가':'안암동','염리동':'아현동','성수동2가':'성수2가제2동','성수동1가':'성수1가제1동','중동':'성산동','노고산동':'서교동','신정동':'서강동','창전동':'서강동','삼선동4가':'삼선동','보문동3가':'보문동','동소문동7가':'동선동','당산동4가':'당산제2동','당산동5가':'당산제2동','당산동':'당산제2동','당산동3가':'당산제1동','당산동1가':'당산제1동','당산동2가':'당산제1동','본동':'노량진동','신수동':'노고산동','대흥동':'노고산동','금호동4가':'금호동','금호동2가':'금호동','충무로4가':'광희동','방화동':'공항동','도화동':'공덕동','신공덕동':'공덕동','일원동':'개포동'}
            def change_dongs(location):
                parts = location.split(' ')
                if parts[2] in old_and_new_dongs:
                    parts[2] = old_and_new_dongs[parts[2]]
                return ' '.join(parts)
            table_1_copy['location'] = table_1_copy['location'].apply(change_dongs)

            # 동 종류
            table_1_copy['district'] = table_1_copy['location'].apply(lambda x: x.split(' ')[2])
            districts = table_1_copy['district'].unique()
        elif SUB == False:
            # 구 종류
            table_1_copy['district'] = table_1_copy['location'].apply(lambda x: x.split(' ')[1])
            districts = table_1_copy['district'].unique()
        else:
            raise ValueError("Invalid value for 'SUB'. It must be either True or False.")
                # (동별) 최대 단지 개수
        # TRAIN: 38 
        # TEST: 24
        max_apartment_complexes = max(table_1_copy.groupby('district')['name'].count())

        # 메모리 효율을 위해 데이터를 인덱스 기반으로 저장
        self.data_indices = []  # (district_idx, time_idx) 튜플 저장
        self.district_data = {}  # district별 전처리된 데이터 저장
        self.max_apartment_complexes = max_apartment_complexes
          # 경제 지표는 공통으로 사용되므로 미리 준비 (CPU에서)
        economy_values = table_2_copy[['call_rate','m2']].values
        self.economy_tensor = torch.FloatTensor(economy_values).to(self.device_cpu)

        if model != 'None': # 임베딩 벡터를 사용할 때
            model.eval()
            # 모델을 GPU에 유지 (원래 DEVICE 사용)
            self.model = model.to(DEVICE)
        else:
            self.model = None

        # 동 마다 전처리하여 district_data에 저장
        for district_idx, district in enumerate(districts): 
            # district별 아파트 단지 정보 추출
            district_apartment_complexes_values = table_1_copy[table_1_copy['district'] == district][[cols for cols in table_1_copy.columns if cols not in ['aid','location','name','district']]].values
            district_apartment_complexes_aids = table_1_copy[table_1_copy['district'] == district]['aid'].values
            
            # district별 가격 정보 전처리 (CPU에서)
            district_apartment_complexes_prices = torch.zeros(district_apartment_complexes_aids.shape[0], len(table_2_copy), 1, device=self.device_cpu)
            for i, district_apartment_complex_aid in enumerate(district_apartment_complexes_aids):
                price_data = pd.DataFrame({'did': range(0, len(table_2_copy))}).merge(
                    table_3_copy[table_3_copy['aid'] == district_apartment_complex_aid][['did','price']], 
                    on='did', how='outer').fillna(0).set_index('did').values
                district_apartment_complexes_prices[i] = torch.from_numpy(price_data)
            
            # district 데이터 저장 (CPU에 유지)
            self.district_data[district] = {
                'apartment_values': district_apartment_complexes_values,
                'prices': district_apartment_complexes_prices,
                'num_apartments': district_apartment_complexes_values.shape[0]
            }
            
            # 각 시점에 대한 인덱스 생성
            for time_idx in range(len(table_2_copy) - window_size):
                self.data_indices.append((district, time_idx))
          # 메모리 정리
        del table_1_copy, table_2_copy, table_3_copy
        gc.collect()

    def _get_embedding_matrix(self, district, time_idx):
        """지연 로딩: 필요할 때만 임베딩 매트릭스 생성"""
        district_data = self.district_data[district]
        district_apartment_complexes_values = district_data['apartment_values']
        
        # 입력 텐서 생성 (CPU에서)
        encoder_input_tensors = torch.zeros(district_apartment_complexes_values.shape[0], 
                                          self.window_size, 12, device=self.device_cpu)
        
        for i, apartment_values in enumerate(district_apartment_complexes_values):
            # 해당 시점의 윈도우 데이터 생성
            apartment_tensor = torch.FloatTensor(apartment_values).repeat(self.window_size, 1)
            economy_window = self.economy_tensor[time_idx:time_idx + self.window_size]
            encoder_input = torch.cat((apartment_tensor, economy_window), dim=1)
            encoder_input_tensors[i] = encoder_input
        
        if self.embedding_dim != 'None' and self.model is not None:
            # 임베딩 생성 (모델은 이미 GPU에 있음)
            with torch.no_grad():
                embedding_matrix = torch.zeros(district_apartment_complexes_values.shape[0], 
                                             self.window_size, self.embedding_dim, device=self.device_cpu)
                
                # 배치 단위로 처리하여 메모리 효율성 증대
                batch_size = 4  # 작은 배치로 처리
                for i in range(0, encoder_input_tensors.shape[0], batch_size):
                    end_idx = min(i + batch_size, encoder_input_tensors.shape[0])
                    batch_input = encoder_input_tensors[i:end_idx].to(self.DEVICE)
                    
                    batch_embeddings = torch.zeros(end_idx - i, self.window_size, self.embedding_dim, device=self.device_cpu)
                    for j in range(end_idx - i):
                        # 모델은 이미 GPU에 있으므로 바로 사용
                        embedding = self.model.encoder(batch_input[j]).cpu()
                        batch_embeddings[j] = embedding
                    
                    embedding_matrix[i:end_idx] = batch_embeddings
                    
                    # GPU 메모리 정리
                    del batch_input, batch_embeddings
                    torch.cuda.empty_cache()
        else:
            embedding_matrix = encoder_input_tensors
        
        return embedding_matrix

    def __getitem__(self, i):
        district, time_idx = self.data_indices[i]
        district_data = self.district_data[district]
        
        # 임베딩 매트릭스 지연 로딩
        embedding_matrix = self._get_embedding_matrix(district, time_idx)
        
        # 결과 텐서 준비 (GPU로 이동)
        if self.embedding_dim == 'None':
            result_embedding = torch.zeros(self.max_apartment_complexes, self.window_size, 12)
        else:
            result_embedding = torch.zeros(self.max_apartment_complexes, self.window_size, self.embedding_dim)
        
        result_prices = torch.zeros(self.max_apartment_complexes, 1)
        
        # 데이터 복사
        num_apartments = district_data['num_apartments']
        result_embedding[:num_apartments] = embedding_matrix
        result_prices[:num_apartments] = district_data['prices'][:, time_idx + self.window_size]
        
        # 인덱스 계산
        price_index = torch.where(result_prices > 0, 1, 0).squeeze()
        
        # GPU로 이동
        result_embedding = result_embedding.to(self.DEVICE)
        result_prices = result_prices.to(self.DEVICE)
        price_index = price_index.to(self.DEVICE)
        
        return result_embedding, num_apartments, price_index, result_prices
    
    def __len__(self):
        return len(self.data_indices)