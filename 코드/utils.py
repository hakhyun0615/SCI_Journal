import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 전처리
def preprocess(transaction_all, economy_all, window_size=5):
    # transaction_all 전처리
    transaction_all['계약연도'] = transaction_all['계약년월'].apply(lambda x: str(x)[:4])
    transaction_all['거래금액(만원)'] = transaction_all['거래금액(만원)'].apply(lambda x: int(x.replace(',','')))
    transaction_all['제곱미터당 거래금액(만원)'] = transaction_all['거래금액(만원)'] / transaction_all['전용면적(㎡)']
    transaction_all['단지'] = transaction_all['시군구'] + ' ' + transaction_all['단지명']
    transaction_all = transaction_all[['단지', '계약년월', '건축년도', '제곱미터당 거래금액(만원)']]

    # economy_all 전처리
    date_range = pd.date_range(start="2006-01", periods=len(economy_all), freq="M")
    economy_all['계약년월'] = date_range.strftime('%Y%m').astype(int)

    # 동일 동,단지이지만 건축년도가 다른 단지 제거
    transaction_all = transaction_all.groupby('단지').filter(lambda x: x['건축년도'].nunique() == 1)

    # 거개량 적은(window_size+1 미만) 단지 제거
    apartment_complex_volume = transaction_all.groupby('단지').agg({'건축년도':'first','제곱미터당 거래금액(만원)':'count'}).reset_index().rename(columns={'제곱미터당 거래금액(만원)':'거래량'})
    small_volume_apartment_complexes = apartment_complex_volume[apartment_complex_volume['거래량'] < window_size+1]['단지'].to_list()
    transaction_all = transaction_all[~transaction_all['단지'].isin(small_volume_apartment_complexes)].reset_index(drop=True)

    # 이상치 제거
    transaction_all['Z 스코어'] = transaction_all.groupby('단지')['제곱미터당 거래금액(만원)'].transform(lambda x: (x-x.mean()) / x.std())
    transaction_all = transaction_all[transaction_all['Z 스코어'].between(-3, 3)]
    transaction_all = transaction_all.drop(columns=['Z 스코어']).reset_index(drop=True)

    # 동일 단지,계약년월일 경우, 제곱미터당 거래금액(만원) 평균
    transaction_all = transaction_all.groupby(['단지','계약년월','건축년도'])['제곱미터당 거래금액(만원)'].mean().to_frame().reset_index()

    # 동과 단지 분리
    transaction_all['동'] = transaction_all['단지'].apply(lambda x: x.split(' ')[:3]).apply(lambda x: ' '.join(x))
    transaction_all['단지'] = transaction_all['단지'].apply(lambda x: x.split(' ')[3:]).apply(lambda x: ' '.join(x))
    transaction_all = transaction_all[['동','단지','건축년도','계약년월','제곱미터당 거래금액(만원)']]

    return transaction_all, economy_all

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
