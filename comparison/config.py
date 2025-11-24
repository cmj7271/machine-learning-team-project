import torch
import os

class Config:
    # 경로 설정
    DATA_PATH = './data'
    RESULT_DIR = './results'
    
    # 하이퍼파라미터
    MAX_LEN = 256
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    N_EMOTIONS = 28
    
    # 🔧 개선 1: 모델별 Dropout 조정
    DROPOUT_MLP = 0.3
    DROPOUT_LSTM = 0.5  # LSTM은 더 높은 dropout 필요
    
    BATCH_SIZE = 32
    EPOCHS = 10
    
    # 🔧 개선 2: 모델별 Learning Rate 분리
    LR_MLP = 2e-3      # MLP는 기존 유지
    LR_LSTM = 5e-4     # LSTM은 낮은 learning rate
    
    # 🔧 개선 3: Gradient Clipping 추가
    GRAD_CLIP = 1.0
    
    # 🔧 개선 4: Learning Rate Scheduler
    USE_SCHEDULER = True
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    
    # 데이터 샘플링
    LABELED_SIZE = 57164
    UNLABELED_SIZE = 37103
    
    # Device
    DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def get_dir(model_name):
        path = os.path.join(Config.RESULT_DIR, model_name)
        os.makedirs(path, exist_ok=True)
        return path