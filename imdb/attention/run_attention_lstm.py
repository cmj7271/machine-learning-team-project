"""
=================================================================
LSTM 모델 학습 곡선 분석 - 실행 스크립트
=================================================================

이 스크립트는 LSTM 모델의 학습 곡선을 분석하고,
MLP 결과와 비교하여 어떤 모델이 더 효과적인지 확인합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json

# 사용자 설정
VOCAB_SIZE = 44851
MAX_LEN = 235
EMBEDDING_DIM = 128

# 파일 경로 (본인의 경로로 수정)
VECTORIZER_PATH = './model/vectorizer_layer_model.keras'
PREPROCESSED_PATH = './data/imdb_preprocessed.csv'

def load_data():
    """데이터 로드 및 전처리"""
    print("="*70)
    print("📂 데이터 로드")
    print("="*70)
    
    # 벡터라이저 로드
    vectorizer = keras.models.load_model(VECTORIZER_PATH)
    print("✅ 토크나이저 로드 완료")
    
    # 데이터 로드
    data = pd.read_csv(PREPROCESSED_PATH)
    print(f"✅ 데이터 로드 완료: {len(data):,} samples")
    
    # 레이블 변환
    label_map = {'positive': 1, 'negative': 0}
    y = data['sentiment'].map(label_map).astype('int32').values
    
    # 텍스트 벡터화
    print("텍스트 벡터화 진행 중... (시간이 걸릴 수 있습니다)")
    X = vectorizer(tf.constant(data['review'].tolist()))
    X = tf.cast(X, tf.int32).numpy()
    
    print(f"✅ 벡터화 완료: X shape = {X.shape}")
    
    return X, y

def run_lstm_experiment(X, y):
    """LSTM 학습 곡선 실험 실행"""
    
    print("\n" + "="*70)
    print("🔬 LSTM 학습 곡선 실험")
    print("="*70)
    
    from attention_lstm import (
        run_learning_curve_experiment,
        plot_learning_curves,
        analyze_results
    )
    
    # 실험할 데이터 크기 (MLP와 동일)
    train_sizes = [500, 1000, 2000, 5000, 10000, 20000, 40000]
    
    print(f"실험 데이터 크기: {train_sizes}")
    print(f"모델: attention + LSTM (32 units)")
    print(f"Regularization: Dropout 0.5, Early Stopping (patience=2)")
    print(f"\n⏱️  예상 소요 시간: 약 1-2시간 (M2 Pro 기준)")
    
    response = input("\n실험을 진행하시겠습니까? (y/n): ")
    
    if response.lower() == 'y':
        # 실험 실행
        print("\n실험 시작... 각 크기별로 최대 10 epochs 학습합니다.")
        print("(Early stopping이 작동하면 더 일찍 종료될 수 있습니다)\n")
        
        lstm_results = run_learning_curve_experiment(X, y, train_sizes)
        
        # 결과 시각화
        print("\n" + "="*70)
        print("📊 결과 시각화")
        print("="*70)
        plot_learning_curves(lstm_results)
        
        # 결과 분석
        analyze_results(lstm_results)
        
        # 결과 저장
        with open('./attention_lstm_results.json', 'w') as f:
            json.dump(lstm_results, f, indent=2)
        print("\n✅ LSTM 결과가 'attention_lstm_results.json'에 저장되었습니다.")
        
        return lstm_results
    else:
        print("⏭️  실험을 건너뜁니다.")
        return None

def compare_models(lstm_results):
    """LSTM과 MLP 결과 비교"""
    
    print("\n" + "="*70)
    print("⚖️  LSTM vs attention + lstm 비교")
    print("="*70)
    
    # MLP 결과 로드
    try:
        with open('advanced_mlp_results.json', 'r') as f:
            mlp_results = json.load(f)
        
        print("✅ LSTM 결과 로드 완료")
        
        # 비교 분석
        from attention_lstm import compare_with_mlp
        compare_with_mlp(lstm_results, mlp_results)
        
    except FileNotFoundError:
        print("⚠️  'advanced_mlp_results.json' 파일을 찾을 수 없습니다.")
        print("   MLP 실험을 먼저 실행하거나, 파일 경로를 확인하세요.")

def main():
    """메인 실행 함수"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║    LSTM 모델 학습 곡선 분석                                   ║
    ║                                                               ║
    ║    이 스크립트는 다음을 수행합니다:                          ║
    ║    1. LSTM 모델의 학습 곡선 분석                             ║
    ║    2. MLP와의 성능 비교                                      ║
    ║    3. 어떤 모델이 더 적합한지 판단                           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # 데이터 로드
        X, y = load_data()
        
        # LSTM 실험 실행
        lstm_results = run_lstm_experiment(X, y)
        
        # MLP와 비교
        if lstm_results is not None:
            compare_models(lstm_results)
        
        print("\n" + "="*70)
        print("✅ 모든 실험이 완료되었습니다!")
        print("="*70)
        print("\n생성된 파일:")
        print("  - advanced_lstm_learning_curves.png: LSTM 학습 곡선")
        print("  - advanced_lstm_results.json: LSTM 수치 데이터")
        print("  - mlp_vs_lstm_comparison.png: MLP vs LSTM 비교 그래프")
        
        print("\n📝 결과 해석 가이드:")
        print("  1. Validation Accuracy가 MLP보다 높다면:")
        print("     → LSTM의 순서 정보 활용이 효과적")
        print("     → Attention 추가로 더 개선 가능")
        print("\n  2. MLP와 비슷하거나 낮다면:")
        print("     → 감성 분석에서는 순서 정보가 덜 중요")
        print("     → 더 간단한 MLP가 더 효율적")
        print("\n  3. 과적합 gap이 크다면:")
        print("     → Dropout 0.6으로 증가")
        print("     → LSTM units 64 → 32로 감소")
        
    except FileNotFoundError as e:
        print(f"\n❌ 파일을 찾을 수 없습니다: {e}")
        print("\n다음 사항을 확인하세요:")
        print("  1. VECTORIZER_PATH와 PREPROCESSED_PATH가 올바른지 확인")
        print("  2. 해당 경로에 파일이 존재하는지 확인")
        print(f"\n현재 설정:")
        print(f"  VECTORIZER_PATH = {VECTORIZER_PATH}")
        print(f"  PREPROCESSED_PATH = {PREPROCESSED_PATH}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 경로 설정 안내
    print("\n⚠️  중요: 파일 경로를 설정하세요!")
    print("="*70)
    print(f"현재 설정된 경로:")
    print(f"  VECTORIZER_PATH = {VECTORIZER_PATH}")
    print(f"  PREPROCESSED_PATH = {PREPROCESSED_PATH}")
    print("\n경로가 올바르지 않다면 스크립트 상단의 경로를 수정하세요.")
    print("="*70)
    
    response = input("\n경로가 올바르면 Enter를 눌러 계속하세요 (종료하려면 q): ")
    
    if response.lower() != 'q':
        main()
    else:
        print("프로그램을 종료합니다.")
