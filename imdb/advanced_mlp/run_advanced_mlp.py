"""
=================================================================
감성 분석 모델 진단 및 개선 실험 - 메인 실행 스크립트
=================================================================

이 스크립트는 학습 곡선 분석과 모델 비교 실험을 순차적으로 실행합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 사용자 설정 (필요시 수정)
VOCAB_SIZE = 44851
MAX_LEN = 235
EMBEDDING_DIM = 128

# 파일 경로 (본인의 경로로 수정)
VECTORIZER_PATH = './model/vectorizer_layer_model.keras'
PREPROCESSED_PATH = './data/imdb_preprocessed.csv'

def load_data():
    """데이터 로드 및 전처리"""
    print("="*70)
    print("1. 데이터 로드 중...")
    print("="*70)
    
    # 벡터라이저 로드
    vectorizer = keras.models.load_model(VECTORIZER_PATH)
    print("✅ 토크나이저 로드 완료")
    
    # 데이터 로드
    data = pd.read_csv(PREPROCESSED_PATH)
    print(f"✅ 데이터 로드 완료: {len(data)} samples")
    
    # 레이블 변환
    label_map = {'positive': 1, 'negative': 0}
    y = data['sentiment'].map(label_map).astype('int32').values
    
    # 텍스트 벡터화
    print("텍스트 벡터화 진행 중... (시간이 걸릴 수 있습니다)")
    X = vectorizer(tf.constant(data['review'].tolist()))
    X = tf.cast(X, tf.int32).numpy()
    
    print(f"✅ 벡터화 완료: X shape = {X.shape}")
    
    return X, y

def run_experiments(X, y):
    """모든 실험 실행"""
    
    # ===============================================
    # Experiment 1: 학습 곡선 분석
    # ===============================================
    print("\n" + "="*70)
    print("2. 학습 곡선 실험 시작")
    print("="*70)
    
    from advanced_mlp import (
        run_learning_curve_experiment,
        plot_learning_curves,
        analyze_results
    )
    
    # 실험할 데이터 크기
    train_sizes = [500, 1000, 2000, 5000, 10000, 20000, 40000]
    
    print(f"실험 데이터 크기: {train_sizes}")
    
    response = input("\n학습 곡선 실험을 진행하시겠습니까? (y/n): ")
    
    if response.lower() == 'y':
        lc_results = run_learning_curve_experiment(X, y, train_sizes)
        
        # 결과 시각화
        plot_learning_curves(lc_results)
        
        # 결과 분석
        analyze_results(lc_results)
        
        # 결과 저장
        import json
        with open('advanced_mlp_results.json', 'w') as f:
            json.dump(lc_results, f, indent=2)
        print("\n✅ 학습 곡선 결과가 'advanced_mlp_results.json'에 저장되었습니다.")
    else:
        print("⏭️  학습 곡선 실험을 건너뜁니다.")


def main():
    """메인 실행 함수"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║    감성 분석 모델 진단 및 개선 실험                           ║
    ║                                                               ║
    ║    이 스크립트는 다음을 수행합니다:                          ║
    ║    1. 학습 곡선 분석 (과적합/데이터 부족 진단)               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # 데이터 로드
        X, y = load_data()
        
        # 실험 실행
        run_experiments(X, y)
        
        print("\n" + "="*70)
        print("✅ 모든 실험이 완료되었습니다!")
        print("="*70)
        print("\n생성된 파일:")
        print("  - learning_curves.png: 학습 곡선 그래프")
        print("  - model_comparison.png: 모델 비교 그래프")
        print("  - learning_curve_results.json: 학습 곡선 수치 데이터")
        print("\n자세한 분석은 'analysis_report.md' 파일을 참조하세요.")
        
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