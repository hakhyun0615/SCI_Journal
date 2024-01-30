import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class Embedding_Dataset(Dataset):
    def __init__(self, table_1, table_2, table_3, DEVICE):
        old_and_new_dongs = {'용산동5가':'한강로동','한강로2가':'한강로동','창동':'창제동','돈암동':'정릉동','거여동':'위례동','문정동':'위례동','장지동':'위례동','문배동':'원효로동','산천동':'원효로동','신창동':'원효로동','원효로1가':'원효로동','화곡동':'우장산동','내발산동':'우장산동','영등포동8가':'영등포동','양평동3가':'양평동','안암동1가':'안암동','염리동':'아현동','성수동2가':'성수2가제2동','성수동1가':'성수1가제1동','중동':'성산동','노고산동':'서교동','신정동':'서강동','창전동':'서강동','삼선동4가':'삼선동','보문동3가':'보문동','동소문동7가':'동선동','당산동4가':'당산제2동','당산동5가':'당산제2동','당산동':'당산제2동','당산동3가':'당산제1동','당산동1가':'당산제1동','당산동2가':'당산제1동','본동':'노량진동','신수동':'노고산동','대흥동':'노고산동','금호동4가':'금호동','금호동2가':'금호동','충무로4가':'광희동','방화동':'공항동','도화동':'공덕동','신공덕동':'공덕동','일원동':'개포동'}
        def change_dongs(location):
            parts = location.split(' ')
            if parts[2] in old_and_new_dongs:
                parts[2] = old_and_new_dongs[parts[2]]
            return ' '.join(parts)
        table_1['location'] = table_1['location'].apply(change_dongs)

        scaler = StandardScaler()
        table_1[[cols for cols in table_1.columns if cols not in ['aid','location','name']]] = scaler.fit_transform(table_1[[cols for cols in table_1.columns if cols not in ['aid','location','name']]])
        scaler.fit(table_2[[cols for cols in table_2.columns if cols not in ['did','year','month']]][:135])
        table_2[[cols for cols in table_2.columns if cols not in ['did','year','month']]] = scaler.transform(table_2[[cols for cols in table_2.columns if cols not in ['did','year','month']]])

        table_merge = pd.merge(table_1, table_3, how='left', on='aid')
        table_merge = pd.merge(table_merge, table_2, how='left', on='did')
        table_merge.sort_values(by='did',inplace=True)

        input_values = table_merge[[cols for cols in table_merge.columns if cols not in ['aid','location','name','did','year','month','price']]].values
        output_values = table_merge[['price']].values * 0.0001

        self.input_tensor = torch.FloatTensor(input_values).to(DEVICE)
        self.output_tensor = torch.FloatTensor(output_values).to(DEVICE)

    def __getitem__(self, i):
        return self.input_tensor[i], self.output_tensor[i]

    def __len__(self):
        return len(self.input_tensor)
