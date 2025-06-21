import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Embedding 학습 및 테스트 코드 (Embedding Training and Testing)")
print("=" * 80)
print(f"학습 데이터: MissingValue/gp_2021.xlsx")
print(f"테스트 데이터: MissingValue/table_merge_2023.xlsx")
print("=" * 80)

# 데이터 로드
print("\n=== 데이터 로드 (Data Loading) ===")
train_data_path = '../../데이터/MissingValue/gp_2021.xlsx'
test_data_path = '../../데이터/MissingValue/table_merge_2023.xlsx'

try:
    train_data = pd.read_excel(train_data_path)
    print(f"✓ 학습 데이터 로드 완료: {train_data.shape}")
    print(f"  컬럼: {train_data.columns.tolist()}")
except Exception as e:
    print(f"✗ 학습 데이터 로드 실패: {e}")
    train_data = None

try:
    test_data = pd.read_excel(test_data_path)
    print(f"✓ 테스트 데이터 로드 완료: {test_data.shape}")
    print(f"  컬럼: {test_data.columns.tolist()}")
except Exception as e:
    print(f"✗ 테스트 데이터 로드 실패: {e}")
    test_data = None

# 데이터가 로드된 경우에만 계속 진행
if train_data is not None and test_data is not None:
    print("\n=== 데이터 기본 정보 ===")
    print("학습 데이터:")
    print(train_data.info())
    print("\n테스트 데이터:")
    print(test_data.info())
    
    print("\n=== 결측값 확인 ===")
    print("학습 데이터 결측값:")
    print(train_data.isnull().sum())
    print("\n테스트 데이터 결측값:")
    print(test_data.isnull().sum())

# Embedding 모델 클래스 정의
class ApartmentEmbeddingModel(nn.Module):
    def __init__(self, categorical_dims, embedding_dims, numerical_features, hidden_dim=512, output_dim=1):
        """
        아파트 데이터용 Embedding 모델
        
        Args:
            categorical_dims: 각 범주형 변수의 고유값 개수 딕셔너리
            embedding_dims: 각 범주형 변수의 임베딩 차원 딕셔너리
            numerical_features: 수치형 변수 개수
            hidden_dim: 은닉층 차원
            output_dim: 출력 차원 (가격 예측이므로 1)
        """
        super(ApartmentEmbeddingModel, self).__init__()
        
        # Embedding 레이어 생성
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        for col_name, num_categories in categorical_dims.items():
            embed_dim = embedding_dims[col_name]
            self.embeddings[col_name] = nn.Embedding(num_categories, embed_dim)
            total_embedding_dim += embed_dim
        
        # 전체 입력 차원 = 임베딩 차원 + 수치형 변수 차원
        total_input_dim = total_embedding_dim + numerical_features
        
        # 신경망 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
    def forward(self, categorical_inputs, numerical_inputs):
        # 각 범주형 변수에 대한 임베딩 계산
        embedded_features = []
        for col_name, cat_input in categorical_inputs.items():
            embedded = self.embeddings[col_name](cat_input)
            embedded_features.append(embedded)
        
        # 모든 임베딩을 연결
        if embedded_features:
            embedded_concat = torch.cat(embedded_features, dim=1)
            # 수치형 변수와 결합
            combined_features = torch.cat([embedded_concat, numerical_inputs], dim=1)
        else:
            combined_features = numerical_inputs
        
        # 신경망을 통한 예측
        output = self.fc_layers(combined_features)
        return output

# 데이터셋 클래스 정의
class ApartmentDataset(Dataset):
    def __init__(self, data, categorical_cols, numerical_cols, target_col, 
                 categorical_encoders=None, numerical_scaler=None, is_train=True):
        """
        아파트 데이터셋 클래스
        
        Args:
            data: 데이터프레임
            categorical_cols: 범주형 변수 리스트
            numerical_cols: 수치형 변수 리스트
            target_col: 타겟 변수명
            categorical_encoders: 범주형 변수 인코더 (None이면 새로 생성)
            numerical_scaler: 수치형 변수 스케일러 (None이면 새로 생성)
            is_train: 학습 데이터 여부
        """
        self.data = data.copy()
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.target_col = target_col
        self.is_train = is_train
        
        # 범주형 변수 인코딩
        if categorical_encoders is None:
            self.categorical_encoders = {}
            for col in categorical_cols:
                if col in self.data.columns:
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col].astype(str))
                    self.categorical_encoders[col] = le
        else:
            self.categorical_encoders = categorical_encoders
            for col in categorical_cols:
                if col in self.data.columns:
                    le = self.categorical_encoders[col]
                    try:
                        self.data[col] = le.transform(self.data[col].astype(str))
                    except ValueError:
                        # 새로운 범주 처리
                        self.data[col] = 0  # 또는 다른 기본값
        
        # 수치형 변수 정규화
        if numerical_scaler is None and is_train:
            self.numerical_scaler = StandardScaler()
            if numerical_cols:
                self.data[numerical_cols] = self.numerical_scaler.fit_transform(
                    self.data[numerical_cols].fillna(0)
                )
        elif numerical_scaler is not None:
            self.numerical_scaler = numerical_scaler
            if numerical_cols:
                self.data[numerical_cols] = self.numerical_scaler.transform(
                    self.data[numerical_cols].fillna(0)
                )
        else:
            self.numerical_scaler = None
            
        # 타겟 변수 처리
        if target_col in self.data.columns:
            self.targets = self.data[target_col].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(self.data))  # 테스트 데이터의 경우
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 범주형 변수
        categorical_data = {}
        for col in self.categorical_cols:
            if col in self.data.columns:
                categorical_data[col] = torch.tensor(self.data.iloc[idx][col], dtype=torch.long)
        
        # 수치형 변수
        numerical_data = []
        for col in self.numerical_cols:
            if col in self.data.columns:
                numerical_data.append(self.data.iloc[idx][col])
        
        numerical_tensor = torch.tensor(numerical_data, dtype=torch.float32) if numerical_data else torch.tensor([], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return categorical_data, numerical_tensor, target

# 데이터 전처리 함수
def preprocess_data(train_data, test_data):
    """
    학습 및 테스트 데이터 전처리
    """
    print("\n=== 데이터 전처리 (Data Preprocessing) ===")
    
    # 공통 컬럼 확인
    common_cols = set(train_data.columns) & set(test_data.columns)
    print(f"공통 컬럼 수: {len(common_cols)}")
    
    # 타겟 변수 확인 (가격 관련 컬럼 찾기)
    price_cols = [col for col in train_data.columns if 'price' in col.lower() or '가격' in col.lower()]
    if price_cols:
        target_col = price_cols[0]
        print(f"타겟 변수: {target_col}")
    else:
        # 기본적으로 마지막 수치형 컬럼을 타겟으로 사용
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        target_col = numeric_cols[-1] if numeric_cols else None
        print(f"기본 타겟 변수: {target_col}")
    
    # 범주형/수치형 변수 분리
    categorical_cols = []
    numerical_cols = []
    
    for col in common_cols:
        if col == target_col:
            continue
            
        if train_data[col].dtype == 'object' or train_data[col].nunique() < 50:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    print(f"범주형 변수 ({len(categorical_cols)}개): {categorical_cols[:10]}...")  # 처음 10개만 출력
    print(f"수치형 변수 ({len(numerical_cols)}개): {numerical_cols[:10]}...")  # 처음 10개만 출력
    
    return categorical_cols, numerical_cols, target_col

# 모델 학습 함수
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    모델 학습
    """
    print(f"\n=== 모델 학습 시작 (Training Started) ===")
    print(f"Epochs: {epochs}, Learning Rate: {lr}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for categorical_data, numerical_data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # 데이터를 GPU로 이동
            categorical_inputs = {k: v.to(device) for k, v in categorical_data.items()}
            numerical_inputs = numerical_data.to(device)
            targets = targets.to(device)
            
            # 순전파
            optimizer.zero_grad()
            outputs = model(categorical_inputs, numerical_inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_count += 1
        
        avg_train_loss = train_loss / train_count
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for categorical_data, numerical_data, targets in val_loader:
                categorical_inputs = {k: v.to(device) for k, v in categorical_data.items()}
                numerical_inputs = numerical_data.to(device)
                targets = targets.to(device)
                
                outputs = model(categorical_inputs, numerical_inputs)
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss += loss.item()
                val_count += 1
        
        avg_val_loss = val_loss / val_count if val_count > 0 else float('inf')
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        # 최고 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    # 최고 모델 로드
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# 모델 평가 함수
def evaluate_model(model, test_loader, test_targets=None):
    """
    모델 평가
    """
    print("\n=== 모델 평가 (Model Evaluation) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for categorical_data, numerical_data, targets in test_loader:
            categorical_inputs = {k: v.to(device) for k, v in categorical_data.items()}
            numerical_inputs = numerical_data.to(device)
            
            outputs = model(categorical_inputs, numerical_inputs)
            predictions.extend(outputs.squeeze().cpu().numpy())
            
            if test_targets is not None:
                actuals.extend(targets.numpy())
    
    predictions = np.array(predictions)
    
    if test_targets is not None:
        actuals = np.array(actuals)
        
        # 평가 메트릭 계산
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # 상관관계
        correlation = np.corrcoef(predictions, actuals)[0, 1]
        print(f"Correlation: {correlation:.4f}")
        
        return predictions, actuals, {'mse': mse, 'rmse': rmse, 'mae': mae, 'correlation': correlation}
    else:
        return predictions, None, None

# 결과 시각화 함수
def plot_results(train_losses, val_losses, predictions=None, actuals=None):
    """
    결과 시각화
    """
    print("\n=== 결과 시각화 (Result Visualization) ===")
    
    # 학습 곡선
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if predictions is not None and actuals is not None:
        # 예측 vs 실제
        plt.subplot(1, 3, 2)
        plt.scatter(actuals, predictions, alpha=0.6, color='purple')
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual')
        plt.grid(True, alpha=0.3)
        
        # 잔차 플롯
        plt.subplot(1, 3, 3)
        residuals = predictions - actuals
        plt.scatter(predictions, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../데이터/Figure/original/Embedding_Training_Results.jpg', dpi=300, bbox_inches='tight')
    plt.show()

# 메인 실행 부분
if train_data is not None and test_data is not None:
    # 데이터 전처리
    categorical_cols, numerical_cols, target_col = preprocess_data(train_data, test_data)
    
    # 실행 계속...
    print("\n✓ 데이터 전처리 완료. 모델 학습을 진행합니다.")
else:
    print("\n✗ 데이터 로드에 실패했습니다. 파일 경로를 확인해주세요.")

# 메인 실행 코드 계속
if train_data is not None and test_data is not None:
    try:
        # 학습/검증 데이터 분할
        from sklearn.model_selection import train_test_split
        
        train_df, val_df = train_test_split(train_data, test_size=0.2, random_state=42)
        print(f"\n학습 데이터: {len(train_df)}개")
        print(f"검증 데이터: {len(val_df)}개")
        print(f"테스트 데이터: {len(test_data)}개")
        
        # 데이터셋 생성
        train_dataset = ApartmentDataset(
            train_df, categorical_cols, numerical_cols, target_col, 
            is_train=True
        )
        
        val_dataset = ApartmentDataset(
            val_df, categorical_cols, numerical_cols, target_col,
            categorical_encoders=train_dataset.categorical_encoders,
            numerical_scaler=train_dataset.numerical_scaler,
            is_train=False
        )
        
        test_dataset = ApartmentDataset(
            test_data, categorical_cols, numerical_cols, target_col,
            categorical_encoders=train_dataset.categorical_encoders,
            numerical_scaler=train_dataset.numerical_scaler,
            is_train=False
        )
        
        # 데이터 로더 생성
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\n데이터 로더 생성 완료 (배치 크기: {batch_size})")
        
        # 임베딩 차원 설정
        categorical_dims = {}
        embedding_dims = {}
        
        for col in categorical_cols:
            if col in train_data.columns:
                unique_count = len(train_dataset.categorical_encoders[col].classes_)
                categorical_dims[col] = unique_count
                # 임베딩 차원은 일반적으로 고유값 개수의 절반 정도로 설정 (최소 4, 최대 50)
                embedding_dims[col] = min(50, max(4, unique_count // 2))
        
        print(f"\n범주형 변수 임베딩 설정:")
        for col in categorical_dims:
            print(f"  {col}: {categorical_dims[col]} → {embedding_dims[col]}차원")
        
        # 모델 생성
        model = ApartmentEmbeddingModel(
            categorical_dims=categorical_dims,
            embedding_dims=embedding_dims,
            numerical_features=len(numerical_cols),
            hidden_dim=512,
            output_dim=1
        )
        
        print(f"\n모델 생성 완료")
        print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        # 모델 학습
        trained_model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, epochs=50, lr=0.001
        )
        
        # 테스트 데이터로 평가
        test_targets = test_data[target_col].values if target_col in test_data.columns else None
        predictions, actuals, metrics = evaluate_model(trained_model, test_loader, test_targets)
        
        # 결과 시각화
        plot_results(train_losses, val_losses, predictions, actuals)
        
        # 모델 저장
        model_save_path = '../../데이터/Checkpoint/embedding/apartment_embedding_model.pth'
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'categorical_dims': categorical_dims,
            'embedding_dims': embedding_dims,
            'categorical_encoders': train_dataset.categorical_encoders,
            'numerical_scaler': train_dataset.numerical_scaler,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'target_col': target_col,
            'metrics': metrics
        }, model_save_path)
        
        print(f"\n모델이 저장되었습니다: {model_save_path}")
        
        # 임베딩 벡터 추출 및 분석
        print("\n=== 임베딩 벡터 분석 (Embedding Vector Analysis) ===")
        
        trained_model.eval()
        embeddings_analysis = {}
        
        for col_name, embedding_layer in trained_model.embeddings.items():
            if col_name in categorical_dims:
                # 해당 범주형 변수의 모든 임베딩 벡터 추출
                num_categories = categorical_dims[col_name]
                embedding_vectors = embedding_layer.weight.data.cpu().numpy()
                
                print(f"\n{col_name} 임베딩:")
                print(f"  - 범주 수: {num_categories}")
                print(f"  - 임베딩 차원: {embedding_vectors.shape[1]}")
                print(f"  - 임베딩 벡터 형태: {embedding_vectors.shape}")
                
                # 임베딩 벡터의 유사도 분석 (상위 5개 범주만)
                if num_categories <= 20:  # 범주가 너무 많지 않은 경우만
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity_matrix = cosine_similarity(embedding_vectors)
                    
                    print(f"  - 임베딩 유사도 (상위 5x5):")
                    print(f"    {similarity_matrix[:5, :5]}")
                
                embeddings_analysis[col_name] = embedding_vectors
        
        # 임베딩 시각화 (차원 축소)
        if len(embeddings_analysis) > 0:
            print("\n=== 임베딩 시각화 (Embedding Visualization) ===")
            
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            
            fig, axes = plt.subplots(2, min(3, len(embeddings_analysis)), figsize=(15, 10))
            if len(embeddings_analysis) == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, (col_name, embeddings) in enumerate(list(embeddings_analysis.items())[:3]):
                if embeddings.shape[0] <= 100:  # 너무 많은 범주는 제외
                    # PCA
                    if embeddings.shape[1] >= 2:
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(embeddings)
                        
                        row_idx = 0
                        col_idx = idx if len(embeddings_analysis) > 1 else 0
                        axes[row_idx, col_idx].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
                        axes[row_idx, col_idx].set_title(f'{col_name} - PCA')
                        axes[row_idx, col_idx].grid(True, alpha=0.3)
                        
                        # t-SNE (데이터가 너무 많지 않은 경우)
                        if embeddings.shape[0] <= 50:
                            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0]-1))
                            tsne_result = tsne.fit_transform(embeddings)
                            
                            row_idx = 1
                            axes[row_idx, col_idx].scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
                            axes[row_idx, col_idx].set_title(f'{col_name} - t-SNE')
                            axes[row_idx, col_idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('../../데이터/Figure/original/Embedding_Visualization.jpg', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 최종 결과 요약
        print("\n" + "="*80)
        print("임베딩 학습 및 테스트 완료 (Embedding Training and Testing Completed)")
        print("="*80)
        
        if metrics:
            print(f"최종 성능 지표:")
            print(f"  - RMSE: {metrics['rmse']:.4f}")
            print(f"  - MAE: {metrics['mae']:.4f}")
            print(f"  - 상관계수: {metrics['correlation']:.4f}")
        
        print(f"\n생성된 파일:")
        print(f"  - 모델 체크포인트: {model_save_path}")
        print(f"  - 학습 결과 그래프: ../../데이터/Figure/original/Embedding_Training_Results.jpg")
        print(f"  - 임베딩 시각화: ../../데이터/Figure/original/Embedding_Visualization.jpg")
        
        print(f"\n임베딩 요약:")
        total_embedding_params = sum(
            categorical_dims[col] * embedding_dims[col] 
            for col in categorical_dims
        )
        print(f"  - 총 임베딩 파라미터: {total_embedding_params:,}")
        print(f"  - 범주형 변수 수: {len(categorical_cols)}")
        print(f"  - 수치형 변수 수: {len(numerical_cols)}")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

else:
    print("\n데이터 로드 실패로 인해 학습을 진행할 수 없습니다.")
    print("다음 사항을 확인해주세요:")
    print("1. 파일 경로가 올바른지 확인")
    print("2. 파일이 존재하는지 확인") 
    print("3. 파일 권한 확인")
    print("4. 파일 형식이 올바른지 확인")

print("\n" + "="*80)
print("스크립트 실행 완료")
print("="*80)
