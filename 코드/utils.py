import numpy as np
import pandas as pd

# 평당가격 선형보간 처리
def price_per_pyeong_interpolate(group):
    idx = pd.date_range(group['계약년월'].min(), group['계약년월'].max(), freq='MS')
    group = group.set_index('계약년월').reindex(idx)
    group['단지명'] = group['단지명'].fillna(method='ffill')
    group['시군구'] = group['시군구'].fillna(method='ffill')
    group['평단가'] = group['평단가'].interpolate()
    return group

# 20060101~20221201 평당가격 결측치 0으로 채우기
def price_per_pyeong_fill_0(df):
    months = pd.to_datetime(pd.date_range(start="20060101", end="20221201", freq='MS'))
    complex_city_combinations = df[['단지명', '시군구']].drop_duplicates()

    combinations = pd.DataFrame({
        '단지명': np.tile(complex_city_combinations['단지명'], len(months)),
        '시군구': np.tile(complex_city_combinations['시군구'], len(months)),
        '계약년월': np.repeat(months, len(complex_city_combinations))
    })
    
    df['계약년월'] = pd.to_datetime(df['계약년월'])
    df = pd.merge(combinations, df, on=['단지명', '계약년월', '시군구'], how='left')
    df['평단가'].fillna(0, inplace=True)

    return df