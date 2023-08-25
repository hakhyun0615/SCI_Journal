import pandas as pd

# 평당가격 선형보간 처리
def price_per_pyeong_interpolate(group):
    idx = pd.date_range(group['계약년월'].min(), group['계약년월'].max(), freq='MS')
    group = group.set_index('계약년월').reindex(idx)
    group['단지명'] = group['단지명'].fillna(method='ffill')
    group['시군구'] = group['시군구'].fillna(method='ffill')
    group['평단가'] = group['평단가'].interpolate()
    return group