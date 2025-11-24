"""
개선된 모델 구성 및 3가지 모델 비교 실험
MLP, LSTM, Attention 모델의 공정한 비교를 위한 코드
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class ImprovedModels:
    """개선된 모델 구성"""
    
    @staticmethod
    def build_mlp(vocab_size=44851, embedding_dim=128, seq_len=235, dropout=0.3):
        """
        개선된 MLP 모델
        - Dropout 추가로 과적합 방지
        """
        model = keras.Sequential([
            layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def build_lstm(vocab_size=44851, embedding_dim=128, seq_len=235, 
                   lstm_units=64, dropout=0.3):
        """
        개선된 LSTM 모델
        - Learning rate 0.001로 조정 (기존 0.01 → 0.001)
        - LSTM units 128 → 64로 감소 (과적합 방지)
        - Dropout 조정
        """
        inputs = keras.Input(shape=(seq_len,), dtype="int32")
        
        x = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True
        )(inputs)
        
        mask = x._keras_mask
        
        # Bidirectional LSTM
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=False)
        )(x, mask=mask)
        
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),  # ✅ 수정!
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    @staticmethod
    def build_lstm_with_attention(vocab_size=44851, embedding_dim=128, 
                                   seq_len=235, lstm_units=64, dropout=0.3):
        """
        개선된 LSTM + Attention 모델
        - Learning rate 조정
        - LSTM units 감소
        """
        # Custom Attention Layer
        class Attention(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.W = layers.Dense(units)
                self.V = layers.Dense(1)

            def call(self, inputs, mask=None):
                score = self.V(tf.nn.tanh(self.W(inputs)))
                score = tf.squeeze(score, axis=-1)

                if mask is not None:
                    score = tf.where(mask, score, tf.fill(tf.shape(score), -1e9))

                attention_weights = tf.nn.softmax(score, axis=1)
                attention_weights = tf.expand_dims(attention_weights, axis=-1)
                context = tf.reduce_sum(attention_weights * inputs, axis=1)

                return context
        
        inputs = keras.Input(shape=(seq_len,), dtype="int32")
        
        x = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True
        )(inputs)
        
        mask = x._keras_mask
        
        # LSTM with return_sequences=True for attention
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True)
        )(x, mask=mask)
        
        # Attention mechanism
        x = Attention(64)(x, mask=mask)
        
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        return model

def compare_models(X_train, X_val, y_train, y_val, epochs=10, batch_size=256):
    """
    3가지 모델을 동일한 조건에서 비교
    
    Returns:
    --------
    results : dict
        각 모델의 학습 히스토리와 최종 성능
    """
    results = {}
    
    models_config = {
        'MLP': ImprovedModels.build_mlp,
        'LSTM': ImprovedModels.build_lstm,
        'LSTM+Attention': ImprovedModels.build_lstm_with_attention
    }
    
    for model_name, build_func in models_config.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name} Model")
        print(f"{'='*60}")
        
        model = build_func()
        model.summary()
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        results[model_name] = {
            'history': history.history,
            'model': model,
            'final_train_acc': history.history['accuracy'][-1],
            'final_val_acc': history.history['val_accuracy'][-1],
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
        }
    
    return results

def print_comparison_table(results):
    """비교 결과를 테이블로 출력"""
    print("\n" + "="*80)
    print("모델 성능 비교 결과")
    print("="*80)
    print(f"{'Model':<20} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<12} {'Val Loss':<12}")
    print("-"*80)
    
    for model_name, result in results.items():
        train_acc = result['final_train_acc']
        val_acc = result['final_val_acc']
        gap = train_acc - val_acc
        val_loss = result['final_val_loss']
        
        print(f"{model_name:<20} {train_acc:<12.4f} {val_acc:<12.4f} {gap:<12.4f} {val_loss:<12.4f}")
    
    print("="*80)
    
    # 최고 성능 모델 찾기
    best_model = max(results.items(), key=lambda x: x[1]['final_val_acc'])
    print(f"\n🏆 최고 성능 모델: {best_model[0]}")
    print(f"   Validation Accuracy: {best_model[1]['final_val_acc']:.4f}")
    
    # 과적합 분석
    print("\n📊 과적합 분석:")
    for model_name, result in results.items():
        gap = result['final_train_acc'] - result['final_val_acc']
        if gap > 0.05:
            print(f"   ❌ {model_name}: 과적합 (gap = {gap:.4f})")
        else:
            print(f"   ✅ {model_name}: 정상 (gap = {gap:.4f})")

def plot_training_histories(results):
    """학습 과정 시각화"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training Accuracy
    for model_name, result in results.items():
        axes[0, 0].plot(result['history']['accuracy'], label=f'{model_name} Train')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Training Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation Accuracy
    for model_name, result in results.items():
        axes[0, 1].plot(result['history']['val_accuracy'], label=f'{model_name} Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Training Loss
    for model_name, result in results.items():
        axes[1, 0].plot(result['history']['loss'], label=f'{model_name} Train')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Validation Loss
    for model_name, result in results.items():
        axes[1, 1].plot(result['history']['val_loss'], label=f'{model_name} Val')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Validation Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('./img/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n그래프가 'model_comparison.png'로 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    """
    사용 방법:
    
    # 1. 데이터 로드 및 전처리
    vectorizer = keras.models.load_model(VECTORIZER_PATH)
    data = pd.read_csv(PREPROCESSED_PATH)
    label_map = {'positive':1, 'negative':0}
    y = data['sentiment'].map(label_map).astype('int32').values
    X = vectorizer(tf.constant(data['review'].tolist()))
    X = tf.cast(X, tf.int32).numpy()
    
    # 2. 데이터 분할
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 3. 모델 비교 실행
    results = compare_models(X_train, X_val, y_train, y_val, epochs=10)
    
    # 4. 결과 출력 및 시각화
    print_comparison_table(results)
    plot_training_histories(results)
    """
    print("모델 비교 실험 모듈이 준비되었습니다.")
    print("위의 주석을 참고하여 실험을 진행하세요.")
