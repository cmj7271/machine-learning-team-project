"""
학습 곡선 분석 실험
데이터 크기를 변화시키며 MLP 모델의 성능 추이 분석
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 전역 변수 설정
VOCAB_SIZE = 44851
MAX_LEN = 235
EMBEDDING_DIM = 128

def build_mlp_model(vocab_size, embedding_dim, seq_len):
    """기본 MLP 모델 생성"""
    model = keras.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),  # 과적합 방지
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def run_learning_curve_experiment(X, y, train_sizes):
    """
    학습 곡선 실험 실행
    
    Parameters:
    -----------
    X : array-like
        전체 입력 데이터
    y : array-like
        전체 레이블
    train_sizes : list
        실험할 훈련 데이터 크기 리스트
        예: [500, 1000, 2000, 5000, 10000, 20000, 40000]
    
    Returns:
    --------
    results : dict
        각 크기별 train/val accuracy와 loss
    """
    results = {
        'train_sizes': [],
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    
    # 전체 데이터를 train/test로 분할 (test는 고정)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    for size in train_sizes:
        print(f"\n{'='*50}")
        print(f"Training with {size} samples")
        print(f"{'='*50}")
        
        # 훈련 데이터를 size만큼만 사용
        if size < len(X_trainval):
            indices = np.random.choice(len(X_trainval), size, replace=False)
            X_train_subset = X_trainval[indices]
            y_train_subset = y_trainval[indices]
        else:
            X_train_subset = X_trainval
            y_train_subset = y_trainval
        
        # 검증 데이터는 항상 동일하게 사용
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_subset, y_train_subset, 
            test_size=0.2, random_state=42, stratify=y_train_subset
        )
        
        # 모델 생성 및 학습
        model = build_mlp_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=256,
            verbose=1
        )
        
        # 결과 저장
        results['train_sizes'].append(size)
        results['train_acc'].append(history.history['accuracy'][-1])
        results['val_acc'].append(history.history['val_accuracy'][-1])
        results['train_loss'].append(history.history['loss'][-1])
        results['val_loss'].append(history.history['val_loss'][-1])
        
        # 중간 결과 출력
        print(f"Train Acc: {results['train_acc'][-1]:.4f}")
        print(f"Val Acc: {results['val_acc'][-1]:.4f}")
        print(f"Gap: {results['train_acc'][-1] - results['val_acc'][-1]:.4f}")
    
    return results

def plot_learning_curves(results):
    """학습 곡선 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy 곡선
    axes[0].plot(results['train_sizes'], results['train_acc'], 'o-', label='Train Accuracy')
    axes[0].plot(results['train_sizes'], results['val_acc'], 'o-', label='Validation Accuracy')
    axes[0].set_xlabel('Training Set Size')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Learning Curve - Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss 곡선
    axes[1].plot(results['train_sizes'], results['train_loss'], 'o-', label='Train Loss')
    axes[1].plot(results['train_sizes'], results['val_loss'], 'o-', label='Validation Loss')
    axes[1].set_xlabel('Training Set Size')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Learning Curve - Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('./img/learning_curves.png', dpi=300, bbox_inches='tight')
    print("\n그래프가 'learning_curves.png'로 저장되었습니다.")
    
    return fig

def analyze_results(results):
    """결과 분석 및 출력"""
    print("\n" + "="*70)
    print("학습 곡선 분석 결과")
    print("="*70)
    
    df = pd.DataFrame(results)
    df['gap'] = df['train_acc'] - df['val_acc']
    
    print(df.to_string(index=False))
    
    print("\n📊 분석:")
    
    # 1. 과적합 여부
    avg_gap = df['gap'].mean()
    if avg_gap > 0.05:
        print(f"❌ 과적합 발생: 평균 gap = {avg_gap:.4f}")
        print("   → Dropout 증가, Regularization 추가 필요")
    else:
        print(f"✅ 과적합 없음: 평균 gap = {avg_gap:.4f}")
    
    # 2. 데이터 충분성
    if len(results['train_sizes']) >= 3:
        last_three_improvements = [
            df['val_acc'].iloc[i+1] - df['val_acc'].iloc[i] 
            for i in range(-3, -1)
        ]
        avg_improvement = np.mean(last_three_improvements)
        
        if avg_improvement > 0.01:
            print(f"📈 데이터 증가시 성능 향상 가능: 최근 평균 개선 = {avg_improvement:.4f}")
            print("   → 더 많은 데이터가 도움될 수 있음")
        else:
            print(f"📊 성능 포화 상태: 최근 평균 개선 = {avg_improvement:.4f}")
            print("   → 데이터보다는 모델 복잡도 조정 필요")
    
    # 3. 권장사항
    print("\n💡 권장사항:")
    if avg_gap > 0.05:
        print("   1. Dropout 비율을 0.3 → 0.5로 증가")
        print("   2. L2 regularization 추가")
        print("   3. Early stopping 도입")
    if df['val_acc'].iloc[-1] < 0.90:
        print("   4. 모델 복잡도 증가 (Dense layer 추가 또는 units 증가)")
        print("   5. 사전 학습된 임베딩 사용 고려")

# 사용 예시
if __name__ == "__main__":
    """
    사용 방법:
    
    # 1. 데이터 로드
    vectorizer = keras.models.load_model(VECTORIZER_PATH)
    data = pd.read_csv(PREPROCESSED_PATH)
    label_map = {'positive':1, 'negative':0}
    y = data['sentiment'].map(label_map).astype('int32').values
    X = vectorizer(tf.constant(data['review'].tolist()))
    X = tf.cast(X, tf.int32).numpy()
    
    # 2. 실험 실행
    train_sizes = [500, 1000, 2000, 5000, 10000, 20000, 40000]
    results = run_learning_curve_experiment(X, y, train_sizes)
    
    # 3. 시각화 및 분석
    plot_learning_curves(results)
    analyze_results(results)
    """
    print("학습 곡선 실험 모듈이 준비되었습니다.")
    print("위의 주석을 참고하여 실험을 진행하세요.")
