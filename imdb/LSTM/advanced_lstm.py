"""
LSTM 학습 곡선 분석 실험
데이터 크기를 변화시키며 LSTM 모델의 성능 추이 분석
MLP와 동일한 조건으로 실험하여 공정한 비교 가능
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

def build_lstm_model(vocab_size, embedding_dim, seq_len, lstm_units=64):
    """
    Bidirectional LSTM 모델 생성
    
    MLP와의 주요 차이점:
    - GlobalAveragePooling 대신 LSTM 사용
    - 단어의 순서 정보를 활용할 수 있음
    - 문맥을 이해하는 능력이 향상됨
    
    Parameters:
    -----------
    vocab_size : int
        어휘 사전 크기
    embedding_dim : int
        임베딩 차원
    seq_len : int
        시퀀스 길이
    lstm_units : int
        LSTM 유닛 수 (기본값 64)
        - 너무 크면 과적합 위험
        - 너무 작으면 표현력 부족
    """
    # Functional API 사용 (masking 처리를 위해)
    inputs = keras.Input(shape=(seq_len,), dtype="int32")
    
    # Embedding layer: 정수 인덱스를 밀집 벡터로 변환
    # mask_zero=True: 패딩(0)을 무시하도록 설정
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True  # 중요: 패딩 토큰 무시
    )(inputs)
    
    # Mask 추출 (패딩 위치 정보)
    # LSTM이 실제 단어만 처리하고 패딩은 무시하도록 함
    mask = x._keras_mask
    
    # Bidirectional LSTM
    # - 양방향으로 문장을 읽어 문맥을 더 잘 이해
    # - return_sequences=False: 마지막 출력만 사용
    # - 64 units: MLP의 32 units보다 조금 크지만, LSTM은 더 복잡한 구조
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False)
    )(x, mask=mask)
    
    # Dense layer: 최종 분류를 위한 fully connected layer
    # 32 units: MLP와 동일하게 유지
    x = layers.Dense(32, activation="relu")(x)
    
    # Dropout: MLP와 동일한 0.5로 과적합 방지
    x = layers.Dropout(0.5)(x)
    
    # Output layer: 이진 분류를 위한 sigmoid
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    # 모델 생성
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # 컴파일: MLP와 동일한 설정
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),  # 표준 learning rate
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def run_learning_curve_experiment(X, y, train_sizes):
    """
    LSTM 학습 곡선 실험 실행
    
    MLP 실험과 동일한 구조로 실행하여 직접 비교 가능
    
    Parameters:
    -----------
    X : array-like
        전체 입력 데이터 (shape: N x seq_len)
    y : array-like
        전체 레이블 (shape: N,)
    train_sizes : list
        실험할 훈련 데이터 크기
        예: [500, 1000, 2000, 5000, 10000, 20000, 40000]
    
    Returns:
    --------
    results : dict
        각 크기별 성능 지표
        - train_sizes: 사용한 데이터 크기
        - train_acc: 훈련 정확도
        - val_acc: 검증 정확도
        - train_loss: 훈련 손실
        - val_loss: 검증 손실
    """
    results = {
        'train_sizes': [],
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    
    # 전체 데이터를 train/test로 분할
    # test는 최종 평가용으로 고정 (여기서는 사용하지 않음)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 각 데이터 크기별로 실험
    for size in train_sizes:
        print(f"\n{'='*50}")
        print(f"Training LSTM with {size} samples")
        print(f"{'='*50}")
        
        # 지정된 크기만큼 데이터 샘플링
        if size < len(X_trainval):
            # 무작위로 size개 선택
            indices = np.random.choice(len(X_trainval), size, replace=False)
            X_train_subset = X_trainval[indices]
            y_train_subset = y_trainval[indices]
        else:
            # 크기가 전체보다 크면 전체 사용
            X_train_subset = X_trainval
            y_train_subset = y_trainval
        
        # Train/Validation 분할
        # 검증 데이터는 항상 20%로 고정
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_subset, y_train_subset, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_train_subset
        )
        
        print(f"실제 사용: Train {len(X_train)}, Val {len(X_val)}")
        
        # LSTM 모델 생성
        model = build_lstm_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN)
        
        # Early Stopping 콜백
        # validation loss가 2 epoch 동안 개선되지 않으면 중단
        # 최고 성능의 가중치로 복원
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            verbose=1
        )
        
        # 모델 학습
        # LSTM은 학습이 느릴 수 있으므로 진행 상황 표시 (verbose=1)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=256,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 결과 저장
        # 마지막 epoch의 성능을 기록
        # (Early stopping이 작동했다면 최고 성능)
        results['train_sizes'].append(size)
        results['train_acc'].append(history.history['accuracy'][-1])
        results['val_acc'].append(history.history['val_accuracy'][-1])
        results['train_loss'].append(history.history['loss'][-1])
        results['val_loss'].append(history.history['val_loss'][-1])
        
        # 중간 결과 출력
        gap = results['train_acc'][-1] - results['val_acc'][-1]
        print(f"\n📊 결과:")
        print(f"  Train Acc: {results['train_acc'][-1]:.4f}")
        print(f"  Val Acc: {results['val_acc'][-1]:.4f}")
        print(f"  Gap: {gap:.4f}")
        
        # 과적합 경고
        if gap > 0.05:
            print(f"  ⚠️  과적합 의심 (Gap > 5%)")
    
    return results

def plot_learning_curves(results, save_path='./img/advanced_lstm_learning_curves.png'):
    """
    LSTM 학습 곡선 시각화
    
    MLP와 동일한 스타일로 그래프 생성하여 비교 용이
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 왼쪽: Accuracy 곡선
    axes[0].plot(results['train_sizes'], results['train_acc'], 
                 'o-', linewidth=2, markersize=8, label='Train Accuracy')
    axes[0].plot(results['train_sizes'], results['val_acc'], 
                 'o-', linewidth=2, markersize=8, label='Validation Accuracy')
    axes[0].set_xlabel('Training Set Size', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('LSTM Learning Curve - Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 오른쪽: Loss 곡선
    axes[1].plot(results['train_sizes'], results['train_loss'], 
                 'o-', linewidth=2, markersize=8, label='Train Loss')
    axes[1].plot(results['train_sizes'], results['val_loss'], 
                 'o-', linewidth=2, markersize=8, label='Validation Loss')
    axes[1].set_xlabel('Training Set Size', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('LSTM Learning Curve - Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 그래프가 '{save_path}'로 저장되었습니다.")
    
    return fig

def analyze_results(results):
    """
    LSTM 학습 곡선 결과 분석
    
    MLP 결과와 비교 가능한 형태로 분석 수행
    """
    print("\n" + "="*70)
    print("LSTM 학습 곡선 분석 결과")
    print("="*70)
    
    # 데이터프레임으로 변환하여 보기 좋게 출력
    df = pd.DataFrame(results)
    df['gap'] = df['train_acc'] - df['val_acc']
    
    # 소수점 4자리까지 표시
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    print("\n" + "="*70)
    print("📊 상세 분석")
    print("="*70)
    
    # 1. 과적합 분석
    avg_gap = df['gap'].mean()
    print(f"\n1️⃣  과적합 여부:")
    if avg_gap > 0.05:
        print(f"   ❌ 과적합 발생: 평균 gap = {avg_gap:.4f}")
        print(f"   → Train과 Val의 차이가 5% 이상입니다.")
        print(f"   → Dropout을 0.6~0.7로 증가하거나 LSTM units를 줄여보세요.")
    else:
        print(f"   ✅ 과적합 없음: 평균 gap = {avg_gap:.4f}")
        print(f"   → 모델이 잘 일반화되고 있습니다.")
    
    # 2. 최종 성능 평가
    final_train_acc = df['train_acc'].iloc[-1]
    final_val_acc = df['val_acc'].iloc[-1]
    final_gap = df['gap'].iloc[-1]
    
    print(f"\n2️⃣  최종 성능 (40,000 샘플 기준):")
    print(f"   Train Accuracy: {final_train_acc:.4f}")
    print(f"   Val Accuracy: {final_val_acc:.4f}")
    print(f"   Gap: {final_gap:.4f}")
    
    # 3. 데이터 충분성 분석
    if len(results['train_sizes']) >= 3:
        # 마지막 3개 구간의 개선도 계산
        last_three_improvements = [
            df['val_acc'].iloc[i+1] - df['val_acc'].iloc[i] 
            for i in range(-3, -1)
        ]
        avg_improvement = np.mean(last_three_improvements)
        
        print(f"\n3️⃣  데이터 충분성:")
        if avg_improvement > 0.01:
            print(f"   📈 데이터 증가 효과 있음: 최근 평균 개선 = {avg_improvement:.4f}")
            print(f"   → 더 많은 데이터가 도움될 수 있습니다.")
        else:
            print(f"   📊 성능 포화 상태: 최근 평균 개선 = {avg_improvement:.4f}")
            print(f"   → 데이터보다는 모델 구조 개선이 필요합니다.")
    
    # 4. MLP 대비 개선도 추정
    print(f"\n4️⃣  MLP와 비교 시 예상 차이:")
    print(f"   LSTM Val Acc: {final_val_acc:.4f}")
    print(f"   (MLP는 약 0.877이었습니다)")
    
    if final_val_acc > 0.877:
        improvement = (final_val_acc - 0.877) * 100
        print(f"   ✅ LSTM이 {improvement:.2f}%p 더 높습니다!")
        print(f"   → 순서 정보를 활용한 효과가 있습니다.")
    elif final_val_acc < 0.877:
        decline = (0.877 - final_val_acc) * 100
        print(f"   ⚠️  LSTM이 {decline:.2f}%p 더 낮습니다.")
        print(f"   → 감성 분석에서는 순서 정보가 덜 중요할 수 있습니다.")
        print(f"   → 또는 LSTM이 과적합되었을 가능성이 있습니다.")
    else:
        print(f"   ≈ 비슷한 성능입니다.")
        print(f"   → 이 태스크에서는 모델 복잡도의 이점이 제한적입니다.")
    
    # 5. 권장사항
    print(f"\n5️⃣  권장사항:")
    if avg_gap > 0.05:
        print(f"   • Dropout을 0.6으로 증가")
        print(f"   • LSTM units를 64에서 32로 감소")
        print(f"   • L2 regularization 추가")
    
    if final_val_acc < 0.90:
        print(f"   • Bidirectional LSTM 유지 (양방향이 중요)")
        print(f"   • Attention mechanism 추가 고려")
        print(f"   • 사전 학습된 임베딩(GloVe) 사용 고려")
    
    print("\n" + "="*70)

def compare_with_mlp(lstm_results, mlp_results):
    """
    LSTM과 MLP 결과를 직접 비교
    
    Parameters:
    -----------
    lstm_results : dict
        LSTM 실험 결과
    mlp_results : dict
        MLP 실험 결과 (advanced_mlp_results.json에서 로드)
    """
    print("\n" + "="*70)
    print("🔍 LSTM vs MLP 비교 분석")
    print("="*70)
    
    # 비교 테이블 생성
    comparison_df = pd.DataFrame({
        'Size': lstm_results['train_sizes'],
        'MLP_Val': mlp_results['val_acc'],
        'LSTM_Val': lstm_results['val_acc'],
        'Difference': [l - m for l, m in zip(lstm_results['val_acc'], mlp_results['val_acc'])]
    })
    
    print("\n📊 Validation Accuracy 비교:")
    print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # 평균 차이 계산
    avg_diff = comparison_df['Difference'].mean()
    print(f"\n평균 차이: {avg_diff:.4f}")
    
    if abs(avg_diff) < 0.01:
        print("→ 두 모델의 성능이 거의 동일합니다.")
        print("→ 감성 분석에서는 순서 정보의 이점이 제한적입니다.")
    elif avg_diff > 0.01:
        print(f"→ LSTM이 평균 {avg_diff*100:.2f}%p 더 높습니다.")
        print("→ 순서 정보를 활용한 효과가 있습니다.")
    else:
        print(f"→ MLP가 평균 {abs(avg_diff)*100:.2f}%p 더 높습니다.")
        print("→ LSTM의 복잡도가 오히려 방해가 되었을 수 있습니다.")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(comparison_df['Size'], comparison_df['MLP_Val'], 
             'o-', linewidth=2, markersize=8, label='MLP')
    plt.plot(comparison_df['Size'], comparison_df['LSTM_Val'], 
             's-', linewidth=2, markersize=8, label='LSTM')
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('MLP vs LSTM: Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./img/mlp_vs_lstm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 비교 그래프가 './img/mlp_vs_lstm_comparison.png'로 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║    LSTM 학습 곡선 실험 모듈                                   ║
    ║                                                               ║
    ║    이 모듈은 MLP와 동일한 방식으로 LSTM의 성능을 분석합니다.  ║
    ║    두 모델의 결과를 직접 비교할 수 있습니다.                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    사용 방법:
    
    # 1. 데이터 로드
    vectorizer = keras.models.load_model(VECTORIZER_PATH)
    data = pd.read_csv(PREPROCESSED_PATH)
    label_map = {'positive':1, 'negative':0}
    y = data['sentiment'].map(label_map).astype('int32').values
    X = vectorizer(tf.constant(data['review'].tolist()))
    X = tf.cast(X, tf.int32).numpy()
    
    # 2. LSTM 실험 실행
    train_sizes = [500, 1000, 2000, 5000, 10000, 20000, 40000]
    lstm_results = run_learning_curve_experiment(X, y, train_sizes)
    
    # 3. 시각화 및 분석
    plot_learning_curves(lstm_results)
    analyze_results(lstm_results)
    
    # 4. MLP와 비교 (선택사항)
    import json
    with open('advanced_mlp_results.json', 'r') as f:
        mlp_results = json.load(f)
    compare_with_mlp(lstm_results, mlp_results)
    """)
    print("\n✅ LSTM 학습 곡선 실험 모듈이 준비되었습니다.")
