"""
attention_lstm.py
Attention Mechanism을 적용한 LSTM 모델 실험 모듈
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 전역 변수 설정 (기존과 동일)
VOCAB_SIZE = 44851
MAX_LEN = 235
EMBEDDING_DIM = 128

class Attention(layers.Layer):
    """
    Attention Layer 커스텀 구현
    
    LSTM의 모든 출력(Hidden States)을 입력으로 받아
    각 시점의 중요도(Weight)를 계산하고, 가중합(Context Vector)을 반환합니다.
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True  # Masking 지원 설정

    def build(self, input_shape):
        # input_shape: (batch_size, seq_len, hidden_dim)
        # Attention Score를 계산하기 위한 가중치와 편향
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", 
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # 1. Score 계산: e = tanh(Wx + b)
        # x shape: (batch, seq_len, hidden_dim) -> e shape: (batch, seq_len, 1)
        e = K.tanh(K.dot(x, self.W) + self.b)
        
        # 2. Attention Weights 계산: a = softmax(e)
        a = K.softmax(e, axis=1)
        
        # 3. Context Vector 계산: c = sum(a * x)
        # 가중치(a)와 입력(x)을 곱한 뒤 시점(seq_len) 축으로 합산
        output = x * a
        
        # Masking 처리 (패딩된 부분은 합산에서 제외하거나 0으로 처리)
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, axis=-1)
            output = output * mask
            
        return K.sum(output, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_attention_lstm_model(vocab_size, embedding_dim, seq_len, lstm_units=64):
    """
    Attention이 추가된 Bidirectional LSTM 모델 생성
    """
    inputs = keras.Input(shape=(seq_len,), dtype="int32")
    
    # 1. Embedding Layer
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True
    )(inputs)
    
    mask = x._keras_mask
    
    # 2. Bidirectional LSTM
    # 중요 변경점: return_sequences=True
    # Attention을 적용하려면 모든 시점의 Hidden State가 필요합니다.
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True)
    )(x)
    
    # 3. Attention Layer
    # (batch, seq_len, hidden_dim*2) -> (batch, hidden_dim*2)
    # 모든 시점의 정보를 중요도에 따라 요약합니다.
    x = Attention()(x)
    
    # 4. Dense & Output
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def run_learning_curve_experiment(X, y, train_sizes):
    """
    기존 코드와 동일한 실험 로직 (모델 빌드 부분만 변경됨)
    """
    results = {
        'train_sizes': [],
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    for size in train_sizes:
        print(f"\n{'='*50}")
        print(f"Training Attention LSTM with {size} samples")
        print(f"{'='*50}")
        
        if size < len(X_trainval):
            indices = np.random.choice(len(X_trainval), size, replace=False)
            X_train_subset = X_trainval[indices]
            y_train_subset = y_trainval[indices]
        else:
            X_train_subset = X_trainval
            y_train_subset = y_trainval
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_subset, y_train_subset, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_train_subset
        )
        
        # 변경: Attention 모델 사용
        model = build_attention_lstm_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN)
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            verbose=1
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=256,
            callbacks=[early_stopping],
            verbose=1
        )
        
        results['train_sizes'].append(size)
        results['train_acc'].append(history.history['accuracy'][-1])
        results['val_acc'].append(history.history['val_accuracy'][-1])
        results['train_loss'].append(history.history['loss'][-1])
        results['val_loss'].append(history.history['val_loss'][-1])
        
        print(f"  Val Acc: {results['val_acc'][-1]:.4f}")
    
    return results

def plot_learning_curves(results, save_path='./img/attention_lstm_learning_curves.png'):
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
    """
    print("\n" + "="*70)
    print("attention + LSTM 학습 곡선 분석 결과")
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
    print(f"   attention + LSTM Val Acc: {final_val_acc:.4f}")
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
