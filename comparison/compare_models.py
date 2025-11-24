import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import Config

def load_result(name):
    path = os.path.join(Config.RESULT_DIR, name, 'result.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def calculate_efficiency_score(f1, time, memory):
    """효율성 점수 계산: 성능/(시간*메모리)"""
    if time == 0 or memory == 0:
        return 0
    return (f1 * 1000) / (time * memory)  # 스케일 조정

def main():
    # 4가지 모델 비교
    models = [
        "MLP_Baseline",           # 1. No attention
        "FullAttention",          # 2. O(n²) Full Attention ← 새로 추가!
        "KDA_Basic",              # 3. O(n) Linear Attention
        "KDA_SemiSupervised"      # 4. O(n) + Semi-supervised
    ]
    
    results = []
    for m in models:
        res = load_result(m)
        if res:
            res['model'] = m
            res['efficiency'] = calculate_efficiency_score(
                res['f1'], 
                res['duration_sec'], 
                res['peak_memory_mb']
            )
            results.append(res)
    
    if not results:
        print("❌ No results found. Run the training scripts first.")
        print("\nRun order:")
        print("1. python run_mlp.py")
        print("2. python run_full_attention.py")
        print("3. python run_kda_basic.py")
        print("4. python run_kda_semi.py")
        return
        
    df = pd.DataFrame(results)
    
    # 테이블 출력
    print("\n" + "="*80)
    print("📊 Model Comparison Table")
    print("="*80)
    display_cols = ['model', 'f1', 'precision', 'recall', 'duration_sec', 'peak_memory_mb', 'efficiency']
    print(df[display_cols].to_string(index=False))
    print("="*80)
    
    # 복잡도 분석
    print("\n⚙️  Attention Complexity Analysis:")
    print("-" * 80)
    complexity_info = {
        "MLP_Baseline": "No Attention (O(1))",
        "FullAttention": "Full Attention (O(n²)) - Baseline for comparison",
        "KDA_Basic": "Linear Attention (O(n)) - Kimi's approach",
        "KDA_SemiSupervised": "Linear Attention (O(n)) + Semi-supervised"
    }
    for model in models:
        if model in complexity_info:
            print(f"  {model:25s}: {complexity_info[model]}")
    print("-" * 80)
    
    # 그래프 생성
    fig = plt.figure(figsize=(20, 12))
    
    # 색상 설정
    colors = ['gray', 'orange', 'skyblue', 'purple']
    model_labels = [m.replace('_', '\n') for m in df['model']]
    
    # 1. F1 Score
    ax1 = plt.subplot(3, 3, 1)
    bars1 = ax1.bar(model_labels, df['f1'], color=colors)
    ax1.set_title('F1 Score (Higher is better)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('F1 Score')
    for i, (bar, val) in enumerate(zip(bars1, df['f1'])):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Training Time
    ax2 = plt.subplot(3, 3, 2)
    bars2 = ax2.bar(model_labels, df['duration_sec'], color=colors)
    ax2.set_title('Training Time (sec) (Lower is better)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Seconds')
    for i, (bar, val) in enumerate(zip(bars2, df['duration_sec'])):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 20, 
                f'{val:.0f}s', ha='center', va='bottom', fontsize=10)
    
    # 3. Memory Usage
    ax3 = plt.subplot(3, 3, 3)
    bars3 = ax3.bar(model_labels, df['peak_memory_mb'], color=colors)
    ax3.set_title('Peak Memory (MB) (Lower is better)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Memory (MB)')
    for i, (bar, val) in enumerate(zip(bars3, df['peak_memory_mb'])):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.3, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Precision
    ax4 = plt.subplot(3, 3, 4)
    ax4.bar(model_labels, df['precision'], color=colors)
    ax4.set_title('Precision', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.0)
    ax4.set_ylabel('Precision')
    
    # 5. Recall
    ax5 = plt.subplot(3, 3, 5)
    ax5.bar(model_labels, df['recall'], color=colors)
    ax5.set_title('Recall', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 1.0)
    ax5.set_ylabel('Recall')
    
    # 6. Efficiency Score
    ax6 = plt.subplot(3, 3, 6)
    bars6 = ax6.bar(model_labels, df['efficiency'], color=colors)
    ax6.set_title('Efficiency Score (F1/Time/Memory)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Efficiency')
    for i, (bar, val) in enumerate(zip(bars6, df['efficiency'])):
        ax6.text(bar.get_x() + bar.get_width()/2, val + 0.001, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 7. F1 vs Time Trade-off
    ax7 = plt.subplot(3, 3, 7)
    for i, row in df.iterrows():
        ax7.scatter(row['duration_sec'], row['f1'], 
                   s=200, color=colors[i], label=row['model'], alpha=0.7)
        ax7.annotate(row['model'], 
                    (row['duration_sec'], row['f1']),
                    xytext=(10, 5), textcoords='offset points', fontsize=9)
    ax7.set_xlabel('Training Time (sec)')
    ax7.set_ylabel('F1 Score')
    ax7.set_title('Performance vs Time Trade-off', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Complexity Comparison (시각화)
    ax8 = plt.subplot(3, 3, 8)
    seq_lengths = np.array([16, 32, 64, 128, 256])
    
    # Theoretical complexity curves
    o1 = np.ones_like(seq_lengths) * 100  # O(1) - constant
    on = seq_lengths * 2  # O(n) - linear
    on2 = (seq_lengths ** 2) / 10  # O(n²) - quadratic
    
    ax8.plot(seq_lengths, o1, 'o-', color='gray', label='MLP: O(1)', linewidth=2)
    ax8.plot(seq_lengths, on, '^-', color='skyblue', label='KDA: O(n)', linewidth=2)
    ax8.plot(seq_lengths, on2, 's-', color='orange', label='Full Attn: O(n²)', linewidth=2)
    ax8.set_xlabel('Sequence Length')
    ax8.set_ylabel('Relative Complexity')
    ax8.set_title('Attention Complexity Growth', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_yscale('log')
    
    # 9. Summary Table (text)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = "🎯 Key Findings:\n\n"
    
    if 'FullAttention' in df['model'].values:
        full_f1 = df[df['model'] == 'FullAttention']['f1'].values[0]
        kda_f1 = df[df['model'] == 'KDA_Basic']['f1'].values[0]
        full_time = df[df['model'] == 'FullAttention']['duration_sec'].values[0]
        kda_time = df[df['model'] == 'KDA_Basic']['duration_sec'].values[0]
        
        f1_diff = ((kda_f1 - full_f1) / full_f1 * 100)
        time_ratio = kda_time / full_time
        
        summary_text += f"Full Attention (O(n²)):\n"
        summary_text += f"  F1 = {full_f1:.3f}\n"
        summary_text += f"  Time = {full_time:.0f}s\n\n"
        
        summary_text += f"KDA Linear (O(n)):\n"
        summary_text += f"  F1 = {kda_f1:.3f} ({f1_diff:+.1f}%)\n"
        summary_text += f"  Time = {kda_time:.0f}s ({time_ratio:.2f}x)\n\n"
        
        if abs(f1_diff) < 10 and time_ratio < 1.5:
            summary_text += "✅ Linear Attention achieves\n"
            summary_text += "   similar performance with\n"
            summary_text += "   better efficiency!\n"
        elif f1_diff < -10:
            summary_text += "⚠️  Linear Attention needs\n"
            summary_text += "   improvement to match\n"
            summary_text += "   Full Attention\n"
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(Config.RESULT_DIR, 'complete_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comprehensive graph saved to {save_path}")
    
    # 결론 출력
    print("\n" + "="*80)
    print("💡 Recommendations:")
    print("="*80)
    
    if len(results) >= 3:
        best_f1_model = df.loc[df['f1'].idxmax(), 'model']
        most_efficient = df.loc[df['efficiency'].idxmax(), 'model']
        
        print(f"  🏆 Best Performance: {best_f1_model}")
        print(f"  ⚡ Most Efficient: {most_efficient}")
        
        if 'FullAttention' in df['model'].values and 'KDA_Basic' in df['model'].values:
            print(f"\n  📊 Linear Attention Value Proposition:")
            print(f"     - If KDA ≈ Full Attention in F1: Use KDA for efficiency!")
            print(f"     - If KDA < Full Attention: Need hyperparameter tuning")
    
    print("="*80)

if __name__ == "__main__":
    main()