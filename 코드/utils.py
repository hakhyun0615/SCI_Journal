import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 평당가격 보간 처리
def price_interpolate(group):
    idx = pd.date_range(group['계약년월'].min(), group['계약년월'].max(), freq='MS')
    group = group.set_index('계약년월').reindex(idx)
    group['동'] = group['동'].fillna(method='ffill')
    group['단지'] = group['단지'].fillna(method='ffill')
    group['건축년도'] = group['건축년도'].fillna(method='ffill')
    imputer = IterativeImputer(max_iter=10, random_state=0)
    group['제곱미터당 거래금액(만원)'] = imputer.fit_transform(group[['제곱미터당 거래금액(만원)']])
    
    return group

# 20060101~20221201 가격 결측치 0으로 채우기
def price_fill_0(df):
    months = pd.to_datetime(pd.date_range(start="20060101", end="20221201", freq='MS'))
    complex_city_combinations = df[['단지', '동']].drop_duplicates()

    combinations = pd.DataFrame({
        '단지': np.tile(complex_city_combinations['단지'], len(months)),
        '동': np.tile(complex_city_combinations['동'], len(months)),
        '계약년월': np.repeat(months, len(complex_city_combinations))
    })
    
    df['계약년월'] = pd.to_datetime(df['계약년월'].astype(str), format='%Y%m')
    df = pd.merge(combinations, df, on=['단지', '계약년월', '동'], how='left')
    df['제곱미터당 거래금액(만원)'].fillna(0, inplace=True)

    return df