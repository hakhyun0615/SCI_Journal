# 완전한 데이터셋에서 보간법 적용을 위한 함수
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 기존 import들 유지
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Gaussian Process imports
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    GP_AVAILABLE = True
except ImportError:
    print("Warning: sklearn.gaussian_process not available")
    GP_AVAILABLE = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

def create_complete_apartment_data_from_merge(table_merge, aid_value, start_year=2019, start_month=1, end_year=2023, end_month=12):
    """
    병합된 데이터에서 특정 아파트에 대해서만 완전한 시계열 데이터 생성
    """
    # 해당 아파트의 기존 데이터 확인
    existing_apartment_data = table_merge[table_merge['aid'] == aid_value]
    
    if len(existing_apartment_data) == 0:
        print(f"Aid {aid_value} not found in merged data")
        return pd.DataFrame()
    
    # 기본 정보 (첫 번째 레코드에서 가져옴, price 제외)
    base_info = existing_apartment_data.iloc[0].to_dict()
    
    # 모든 날짜에 대해 데이터 생성
    complete_data = []
    current_year = start_year
    current_month = start_month
    
    while current_year < end_year or (current_year == end_year and current_month <= end_month):
        # 해당 날짜의 실제 데이터 확인
        actual_data = existing_apartment_data[
            (existing_apartment_data['year'] == current_year) & 
            (existing_apartment_data['month'] == current_month)
        ]
        
        if len(actual_data) > 0:
            # 실제 데이터가 있으면 그대로 사용
            complete_data.append(actual_data.iloc[0].to_dict())
        else:
            # 실제 데이터가 없으면 기본 정보 + 결측 가격으로 생성
            row_data = base_info.copy()
            row_data['year'] = current_year
            row_data['month'] = current_month
            row_data['price'] = np.nan
            complete_data.append(row_data)
        
        # 다음 달로 이동
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    # 데이터프레임으로 변환
    result_df = pd.DataFrame(complete_data)
    result_df = result_df.sort_values(['year', 'month']).reset_index(drop=True)
    
    return result_df

def apply_interpolation_methods_complete(table_merge, aid_value, methods=['all']):
    """
    병합 데이터 기반 완전한 시계열에서 특정 아파트에 대해 다양한 보간법을 적용
    
    Args:
        aid_value: 아파트 ID
        methods: 적용할 방법 리스트 (기본값: ['all'])
    Returns:
        dict: 각 방법별 결과를 담은 딕셔너리
    """
    # 완전한 시계열 데이터 생성
    apartment_data = create_complete_apartment_data_from_merge(table_merge, aid_value)
    
    if len(apartment_data) == 0:
        print(f"No data found for aid={aid_value}")
        return {}
    
    # time_numeric 추가
    apartment_data['time_numeric'] = (apartment_data['year'] - 2019) * 12 + (apartment_data['month'] - 1)
    
    print(f"\n=== AID {aid_value} 완전한 데이터 보간 결과 ===")
    print(f"전체 데이터 포인트: {len(apartment_data)}")
    print(f"결측치 개수: {apartment_data['price'].isna().sum()}")
    print(f"데이터가 있는 포인트: {apartment_data['price'].notna().sum()}")
    
    results = {}
    
    # 원본 데이터도 저장 (비교용)
    results['original'] = apartment_data.copy()
    
    method_functions = {
        'linear': linear_interpolation,
        'polynomial': polynomial_interpolation,
        'spline': spline_interpolation,
        'ffill': forward_fill,
        'bfill': backward_fill,
        'nearest': nearest_neighbor_interpolation,
        'gp': gaussian_process_interpolation,
        'neural_ode': neural_ode_interpolation  # 항상 포함 (내부적으로 linear로 대체)
    }
    
    if 'all' in methods:
        methods = list(method_functions.keys())
    
    for method in methods:
        if method in method_functions:
            try:
                print(f"\n적용 중: {method}")
                results[method] = method_functions[method](apartment_data.copy())
                remaining_na = results[method]['price'].isna().sum()
                print(f"{method} 완료 - 남은 결측치: {remaining_na}")
            except Exception as e:
                print(f"Error in {method}: {str(e)}")
                results[method] = None
        else:
            print(f"Unknown method: {method}")
    
    return results



# 1. 선형 보간법 (Linear Interpolation)
def linear_interpolation(apartment_data):
    """선형 보간법을 사용하여 결측치 채우기"""
    data = apartment_data.copy()
    
    # 시간순으로 정렬
    data = data.sort_values('time_numeric')
    
    # 결측치가 아닌 데이터만 추출
    valid_data = data.dropna(subset=['price'])
    
    if len(valid_data) < 2:
        print(f"Warning: Not enough valid data points for linear interpolation")
        return data
    
    # 선형 보간 함수 생성
    from scipy.interpolate import interp1d
    f_linear = interp1d(valid_data['time_numeric'], valid_data['price'], 
                       kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # 결측치에 대해 보간값 계산
    missing_idx = data['price'].isna()
    if missing_idx.any():
        data.loc[missing_idx, 'price'] = f_linear(data.loc[missing_idx, 'time_numeric'])
    
    return data

# 2. 다항 보간법 (Polynomial Interpolation)
def polynomial_interpolation(apartment_data, degree=3):
    """다항 보간법을 사용하여 결측치 채우기"""
    data = apartment_data.copy()
    data = data.sort_values('time_numeric')
    
    valid_data = data.dropna(subset=['price'])
    
    if len(valid_data) < degree + 1:
        print(f"Warning: Not enough data points for degree {degree} polynomial")
        return linear_interpolation(data)  # fallback to linear
    
    # 라그랑주 다항식 보간
    from scipy.interpolate import lagrange
    x_valid = valid_data['time_numeric'].values
    y_valid = valid_data['price'].values
      # 다항식 계수 구하기
    poly = lagrange(x_valid, y_valid)
    
    # 다항식 계수 정보 출력
    print(f"다항식 계수 (차수 {len(poly.coef)-1}):")
    print(f"전체 다항식: {poly}")
    print(f"계수 배열: {poly.coef}")
    print(f"최고차항부터 상수항까지: {[f'x^{len(poly.coef)-1-i}: {coef:.6f}' for i, coef in enumerate(poly.coef)]}")
    print("-" * 50)

    # 결측치에 대해 보간값 계산
    missing_idx = data['price'].isna()
    if missing_idx.any():
        data.loc[missing_idx, 'price'] = poly(data.loc[missing_idx, 'time_numeric'])
    
    return data

# 3. 스플라인 보간법 (Spline Interpolation)
def spline_interpolation(apartment_data):
    """스플라인 보간법을 사용하여 결측치 채우기"""
    data = apartment_data.copy()
    data = data.sort_values('time_numeric')
    
    valid_data = data.dropna(subset=['price'])
    
    if len(valid_data) < 4:
        print(f"Warning: Not enough data points for cubic spline")
        return linear_interpolation(data)  # fallback to linear
    
    # 큐빅 스플라인 보간
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(valid_data['time_numeric'], valid_data['price'])
    
    # 결측치에 대해 보간값 계산
    missing_idx = data['price'].isna()
    if missing_idx.any():
        data.loc[missing_idx, 'price'] = cs(data.loc[missing_idx, 'time_numeric'])
    
    return data

# 4. Forward Fill (ffill)
def forward_fill(apartment_data):
    """Forward Fill을 사용하여 결측치 채우기"""
    data = apartment_data.copy()
    data = data.sort_values('time_numeric')
    data['price'] = data['price'].fillna(method='ffill')
    return data

# 5. Backward Fill (bfill)
def backward_fill(apartment_data):
    """Backward Fill을 사용하여 결측치 채우기"""
    data = apartment_data.copy()
    data = data.sort_values('time_numeric')
    data['price'] = data['price'].fillna(method='bfill')
    return data

# 6. Nearest-Neighbor Interpolation
def nearest_neighbor_interpolation(apartment_data):
    """최근접 이웃 보간법을 사용하여 결측치 채우기"""
    data = apartment_data.copy()
    data = data.sort_values('time_numeric')
    
    valid_data = data.dropna(subset=['price'])
    
    if len(valid_data) < 1:
        print(f"Warning: No valid data points for nearest neighbor")
        return data
    
    # 최근접 이웃 보간
    from scipy.interpolate import interp1d
    f_nearest = interp1d(valid_data['time_numeric'], valid_data['price'], 
                        kind='nearest', bounds_error=False, fill_value='extrapolate')
    
    missing_idx = data['price'].isna()
    if missing_idx.any():
        data.loc[missing_idx, 'price'] = f_nearest(data.loc[missing_idx, 'time_numeric'])
    
    return data

# 7. Gaussian Process (GP) 보간법
def gaussian_process_interpolation(apartment_data, length_scale=1.0, noise_level=0.1):
    """가우시안 프로세스를 사용하여 결측치 채우기"""
    data = apartment_data.copy()
    data = data.sort_values('time_numeric')
    
    valid_data = data.dropna(subset=['price'])
    
    if len(valid_data) < 2:
        print(f"Warning: Not enough data points for GP")
        return linear_interpolation(data)
    
    # 가우시안 프로세스 모델 설정
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_level**2, n_restarts_optimizer=10)
    
    # 훈련
    X_train = valid_data['time_numeric'].values.reshape(-1, 1)
    y_train = valid_data['price'].values
    gp.fit(X_train, y_train)
    
    # 예측
    missing_idx = data['price'].isna()
    if missing_idx.any():
        X_pred = data.loc[missing_idx, 'time_numeric'].values.reshape(-1, 1)
        y_pred, sigma = gp.predict(X_pred, return_std=True)
        data.loc[missing_idx, 'price'] = y_pred
    
    return data

# 8. Neural ODE 보간법 (placeholder - torchdiffeq가 없으면 linear로 대체)
def neural_ode_interpolation(apartment_data):
    """Neural ODE를 사용하여 결측치 채우기 (라이브러리 없으면 linear 사용)"""
    print("Neural ODE not available, using linear interpolation instead")
    return linear_interpolation(apartment_data)



# 7. Gaussian Process (GP) 보간법
def gaussian_process_interpolation(apartment_data, length_scale=1.0, noise_level=0.1):
    """가우시안 프로세스를 사용하여 결측치 채우기"""
    if not GP_AVAILABLE:
        print("Warning: Gaussian Process not available, using linear interpolation")
        return linear_interpolation(apartment_data)
        
    data = apartment_data.copy()
    data = data.sort_values('time_numeric')
    
    valid_data = data.dropna(subset=['price'])
    
    if len(valid_data) < 2:
        print(f"Warning: Not enough data points for GP")
        return linear_interpolation(data)
    
    try:
        # 가우시안 프로세스 모델 설정
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_level**2, n_restarts_optimizer=10)
        
        # 훈련
        X_train = valid_data['time_numeric'].values.reshape(-1, 1)
        y_train = valid_data['price'].values
        gp.fit(X_train, y_train)
        
        # GP 모델 정보 출력
        print(f"GP 커널 파라미터:")
        print(f"  - 커널: {gp.kernel_}")
        print(f"  - 로그 우도: {gp.log_marginal_likelihood():.6f}")
        print(f"  - 훈련 데이터 포인트: {len(X_train)}")
        print("-" * 50)
        
        # 예측
        missing_idx = data['price'].isna()
        if missing_idx.any():
            X_pred = data.loc[missing_idx, 'time_numeric'].values.reshape(-1, 1)
            y_pred, sigma = gp.predict(X_pred, return_std=True)
            data.loc[missing_idx, 'price'] = y_pred
            
            # 예측 불확실성 정보
            print(f"GP 예측 정보:")
            print(f"  - 예측 포인트 수: {len(X_pred)}")
            print(f"  - 평균 예측 불확실성 (σ): {np.mean(sigma):.6f}")
            print(f"  - 최대 예측 불확실성 (σ): {np.max(sigma):.6f}")
        
        return data
        
    except Exception as e:
        print(f"Warning: GP failed ({str(e)}), using linear interpolation")
        return linear_interpolation(data)


# 8. Neural ODE 보간법
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, t, y):
        return self.net(y)

def neural_ode_interpolation(apartment_data, hidden_dim=64, epochs=100, lr=0.01):
    """Neural ODE를 사용하여 결측치 채우기"""
    if not NEURAL_ODE_AVAILABLE:
        print("Neural ODE not available, using linear interpolation instead")
        return linear_interpolation(apartment_data)
    
    data = apartment_data.copy()
    data = data.sort_values('time_numeric')
    
    valid_data = data.dropna(subset=['price'])
    
    if len(valid_data) < 3:
        print(f"Warning: Not enough data points for Neural ODE")
        return linear_interpolation(data)
    
    # 데이터 준비
    t_valid = torch.tensor(valid_data['time_numeric'].values, dtype=torch.float32).to(DEVICE)
    y_valid = torch.tensor(valid_data['price'].values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    # 모델 및 옵티마이저 설정
    ode_func = ODEFunc(hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=lr)
    
    # 훈련
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 초기 조건 설정 (첫 번째 유효한 데이터 포인트)
        y0 = y_valid[0:1]
        t_eval = t_valid
        
        # ODE 솔버로 궤적 예측
        pred_y = odeint(ode_func, y0, t_eval)
        
        # 손실 계산 (예측값과 실제값 비교)
        loss = torch.mean((pred_y.squeeze() - y_valid.squeeze())**2)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Neural ODE Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    # 결측치 예측
    missing_idx = data['price'].isna()
    if missing_idx.any():
        t_missing = torch.tensor(data.loc[missing_idx, 'time_numeric'].values, dtype=torch.float32).to(DEVICE)
        
        # 전체 시간에 대해 예측
        t_all = torch.tensor(data['time_numeric'].values, dtype=torch.float32).to(DEVICE)
        y0 = y_valid[0:1]
        
        with torch.no_grad():
            pred_all = odeint(ode_func, y0, t_all)
        
        # 결측치 위치에 예측값 할당
        missing_positions = data.index[missing_idx]
        data_positions = data.index
        
        for pos in missing_positions:
            idx_in_data = list(data_positions).index(pos)
            data.loc[pos, 'price'] = pred_all[idx_in_data].cpu().numpy()[0]
    
    return data

def get_polynomial_coefficients(apartment_data, degree=3):
    """다항 보간법의 계수를 반환하는 함수 (보간 수행 없이)"""
    data = apartment_data.copy()
    data = data.sort_values('time_numeric')
    
    valid_data = data.dropna(subset=['price'])
    
    if len(valid_data) < degree + 1:
        print(f"Warning: Not enough data points for degree {degree} polynomial")
        return None
    
    # 라그랑주 다항식 보간
    from scipy.interpolate import lagrange
    x_valid = valid_data['time_numeric'].values
    y_valid = valid_data['price'].values
    
    # 다항식 계수 구하기
    poly = lagrange(x_valid, y_valid)
    
    return {
        'polynomial': poly,
        'coefficients': poly.coef,
        'degree': len(poly.coef) - 1,
        'x_points': x_valid,
        'y_points': y_valid
    }