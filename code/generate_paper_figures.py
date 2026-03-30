"""
生成论文图表

包括:
1. 训练损失曲线对比 (随机特征 vs ResNet 特征)
2. SR/SPL 指标对比图
3. 方法架构图 (ASCII/TikZ 格式)
4. 消融实验结果图
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
OUTPUT_DIR = Path("/Users/tyrion/Projects/Papers/paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_training_history():
    """加载训练历史"""
    # 随机特征训练历史
    random_history_file = "/Users/tyrion/Projects/Papers/checkpoints/training_history.json"
    with open(random_history_file, 'r') as f:
        random_history = json.load(f)

    # ResNet 特征训练历史
    resnet_history_file = "/Users/tyrion/Projects/Papers/checkpoints/training_history_r2r.json"
    with open(resnet_history_file, 'r') as f:
        resnet_history = json.load(f)

    return random_history, resnet_history


def load_evaluation_results():
    """加载评估结果"""
    eval_file = "/Users/tyrion/Projects/Papers/data/evaluation_r2r/model_evaluation_metrics.json"
    with open(eval_file, 'r') as f:
        return json.load(f)


def plot_training_loss_comparison():
    """Figure 1: 训练损失曲线对比"""
    random_history, resnet_history = load_training_history()

    fig, ax = plt.subplots(figsize=(8, 5))

    # 随机特征
    random_epochs = [h['epoch'] for h in random_history]
    random_loss = [h['val_loss'] for h in random_history]

    # ResNet 特征
    resnet_epochs = [h['epoch'] for h in resnet_history]
    resnet_loss = [h['val_loss'] for h in resnet_history]

    ax.plot(random_epochs, random_loss, 'o--', color='gray', linewidth=2,
            markersize=6, label='Random Features')
    ax.plot(resnet_epochs, resnet_loss, 's-', color='#1f77b4', linewidth=2,
            markersize=6, label='ResNet-50 Features (Ours)')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2.5, 3.7)

    # 标注关键点
    ax.annotate('Best: 3.37', xy=(20, 3.37), xytext=(12, 3.65),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('Best: 2.63', xy=(15, 2.63), xytext=(8, 2.8),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='#1f77b4'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_loss_comparison.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'training_loss_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'training_loss_comparison.png'}")
    plt.close()


def plot_sr_spl_comparison():
    """Figure 2: SR/SPL 指标对比"""
    eval_metrics = load_evaluation_results()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # SR 对比
    models = ['VLN-BERT\n(Paper)', 'Ours\n(ResNet)']
    sr_values = [55, eval_metrics['SR'] * 100]
    colors = ['gray', '#1f77b4']

    bars1 = axes[0].bar(models, sr_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Success Rate (%)', fontsize=11)
    axes[0].set_title('Success Rate Comparison', fontsize=12)
    axes[0].set_ylim(0, 80)

    # 标注数值
    for bar, val in zip(bars1, sr_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11)

    # SPL 对比
    spl_values = [50, eval_metrics['SPL'] * 100]
    bars2 = axes[1].bar(models, spl_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('SPL (%)', fontsize=11)
    axes[1].set_title('SPL Comparison', fontsize=12)
    axes[1].set_ylim(0, 80)

    # 标注数值
    for bar, val in zip(bars2, spl_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sr_spl_comparison.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'sr_spl_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'sr_spl_comparison.png'}")
    plt.close()


def plot_instruction_quality():
    """Figure 3: 指令质量对比 (LLM vs MT)"""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 优秀率对比
    methods = ['Machine\nTranslation', 'LLM\n(Ours)']
    excellent_rates = [20.0, 76.7]
    colors = ['gray', '#2ca02c']

    bars1 = axes[0].bar(methods, excellent_rates, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Excellent Rate (%)', fontsize=11)
    axes[0].set_title('Instruction Quality: Excellent Rate', fontsize=12)
    axes[0].set_ylim(0, 100)

    for bar, val in zip(bars1, excellent_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11)

    # 平均分数对比
    avg_scores = [4.28, 4.61]
    bars2 = axes[1].bar(methods, avg_scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Average Score (1-5)', fontsize=11)
    axes[1].set_title('Instruction Quality: Average Score', fontsize=12)
    axes[1].set_ylim(0, 5)

    for bar, val in zip(bars2, avg_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'instruction_quality.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'instruction_quality.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'instruction_quality.png'}")
    plt.close()


def plot_ablation_study():
    """Figure 4: 消融实验结果"""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 特征类型对比
    features = ['Random\nGaussian', 'ResNet-50\n(Ours)']
    val_losses = [3.37, 2.63]
    colors = ['gray', '#ff7f0e']

    bars1 = axes[0].bar(features, val_losses, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Best Validation Loss', fontsize=11)
    axes[0].set_title('Feature Ablation: Validation Loss', fontsize=12)
    axes[0].set_ylim(0, 4)

    # 标注改进百分比
    improvement = (3.37 - 2.63) / 3.37 * 100
    for bar, val in zip(bars1, val_losses):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    axes[0].text(0.5, 3.5, f'-{improvement:.1f}%', ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.3))

    # 训练时间对比
    train_times = [6, 1]
    bars2 = axes[1].bar(features, train_times, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Training Time (min)', fontsize=11)
    axes[1].set_title('Feature Ablation: Training Time', fontsize=12)
    axes[1].set_ylim(0, 8)

    time_improvement = (6 - 1) / 6 * 100
    for bar, val in zip(bars2, train_times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val} min', ha='center', va='bottom', fontsize=11)
    axes[1].text(0.5, 7, f'-{time_improvement:.1f}%', ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.3))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_study.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'ablation_study.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'ablation_study.png'}")
    plt.close()


def generate_architecture_diagram():
    """Figure 5: 方法架构图 (TikZ 格式)"""

    tikz_code = r"""
\begin{figure*}[t]
\centering
\begin{tikzpicture}[
    node distance=1.5cm,
    block/.style={rectangle, draw, rounded corners, minimum height=1cm, align=center, fill=blue!10},
    arrow/.style={thick,->,>=stealth}
]

% Input
\node (input) [block, fill=green!10] {
    \textbf{Input}\\
    Path Description\\
    (landmarks, directions)
};

% LLM
\node (llm) [block, above of=input, minimum width=4cm, fill=yellow!20] {
    \textbf{LLM (Qwen-3.5)}\\
    Prompt-based Generation
};

% Chinese Instruction
\node (instruction) [block, above of=llm, fill=green!10] {
    \textbf{Chinese Instruction}\\
    "直走 3 米，然后左转..."
};

% VLN Model
\node (vln) [block, right of=instruction, xshift=4cm, minimum width=5cm, fill=blue!20] {
    \textbf{VLN-BERT Baseline}\\
    \small Instruction Encoder + Visual Encoder\\
    \small Cross-Modal Attention
};

% Visual Features
\node (visual) [block, below of=vln, fill=orange!10] {
    \textbf{ResNet-50}\\
    Pre-trained Features\\
    (2048-d $\rightarrow$ 256-d)
};

% Output
\node (output) [block, right of=vln, xshift=3cm, fill=green!10] {
    \textbf{Action Prediction}\\
    Softmax over 36 candidates
};

% Arrows
\draw [arrow] (input) -- node[right] {Path Info} (llm);
\draw [arrow] (llm) -- node[right] {Generate} (instruction);
\draw [arrow] (instruction) -- node[above] {Tokenize} (vln);
\draw [arrow] (visual) -- node[right] {Features} (vln);
\draw [arrow] (vln) -- node[above] {Predict} (output);

\end{tikzpicture}
\caption{Overview of our LLM4VLM framework. The LLM generates Chinese navigation instructions from path descriptions, which are then fed into the VLN-BERT baseline model along with ResNet-50 visual features for action prediction.}
\label{fig:architecture}
\end{figure*}
"""

    # 保存 TikZ 代码
    tikz_file = OUTPUT_DIR / 'architecture_tikz.tex'
    with open(tikz_file, 'w') as f:
        f.write(tikz_code)
    print(f"Saved: {tikz_file}")

    # 同时生成 ASCII 版本
    ascii_art = """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        LLM4VLM Framework Overview                        │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │  Path Input      │
    │  [start]→Living  │
    │  →Hall→Kitchen   │
    └─────────┬────────┘
              │ Path Description
              ▼
    ┌──────────────────┐
    │   LLM Generator  │
    │  (Qwen-3.5-Plus) │
    │  Prompt-based    │
    └─────────┬────────┘
              │ Generate
              ▼
    ┌──────────────────┐
    │ Chinese Instr.   │
    │ "直走 3 米，然后左转"│
    └─────────┬────────┘
              │ Tokenize
              ▼
    ┌─────────────────────────────────────┐
    │         VLN-BERT Baseline           │
    │  ┌─────────────┐  ┌──────────────┐  │
    │  │ Instruction │  │   Visual     │  │
    │  │  Encoder    │  │  Encoder     │  │
    │  │  (256-d)    │  │  (ResNet-50) │  │
    │  └──────┬──────┘  └──────┬───────┘  │
    │         └───────┬────────┘          │
    │         Cross-Modal Attention       │
    └─────────────────┬───────────────────┘
                      │ Predict
                      ▼
    ┌──────────────────┐
    │  Action Output   │
    │  Softmax(36)     │
    │  Best: Turn Left │
    └──────────────────┘
    """

    ascii_file = OUTPUT_DIR / 'architecture_ascii.txt'
    with open(ascii_file, 'w') as f:
        f.write(ascii_art)
    print(f"Saved: {ascii_file}")


def plot_confidence_distribution():
    """Figure 6: 置信度分布图"""

    # 加载详细结果
    detailed_file = "/Users/tyrion/Projects/Papers/data/evaluation_r2r/model_evaluation_detailed.json"
    with open(detailed_file, 'r') as f:
        results = json.load(f)

    confidences = [r['confidence'] for r in results]
    distances = [r['distance_to_goal'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 置信度分布
    axes[0].hist(confidences, bins=20, color='#1f77b4', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Prediction Confidence', fontsize=11)
    axes[0].set_ylabel('Number of Samples', fontsize=11)
    axes[0].set_title('Confidence Distribution', fontsize=12)
    axes[0].axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.4f}')
    axes[0].legend()

    # 距离误差分布
    axes[1].hist(distances, bins=20, color='#ff7f0e', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Distance to Goal (m)', fontsize=11)
    axes[1].set_ylabel('Number of Samples', fontsize=11)
    axes[1].set_title('Distance Error Distribution', fontsize=12)
    axes[1].axvline(3.0, color='red', linestyle='--', label='Success Threshold (3m)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confidence_distribution.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'confidence_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'confidence_distribution.png'}")
    plt.close()


def generate_all_tables():
    """生成论文表格 (LaTeX 格式)"""

    tables = r"""
% Table 1: Main Results
\begin{table}[t]
\centering
\begin{tabular}{lccc}
\hline
\textbf{Model} & \textbf{SR (\%)} & \textbf{SPL (\%)} & \textbf{Oracle SR (\%)} \\
\hline
VLN-BERT (Paper) & ~55 & ~50 & - \\
\textbf{Ours (LLM+ResNet)} & \textbf{62.0} & \textbf{61.8} & \textbf{69.0} \\
\hline
\end{tabular}
\caption{Main results comparison with VLN-BERT baseline.}
\label{tab:main_results}
\end{table}

% Table 2: Instruction Quality Comparison
\begin{table}[t]
\centering
\begin{tabular}{lcc}
\hline
\textbf{Method} & \textbf{Excellent Rate (\%)} & \textbf{Avg Score (1-5)} \\
\hline
Machine Translation & 20.0 & 4.28 \\
\textbf{LLM Direct (Ours)} & \textbf{76.7} & \textbf{4.61} \\
\hline
\end{tabular}
\caption{Instruction quality comparison between machine translation and LLM direct generation.}
\label{tab:instruction_quality}
\end{table}

% Table 3: Feature Ablation Study
\begin{table}[t]
\centering
\begin{tabular}{lcc}
\hline
\textbf{Feature Type} & \textbf{Best Val Loss} & \textbf{Training Time} \\
\hline
Random Gaussian & 3.37 & 6 min \\
\textbf{ResNet-50 (Ours)} & \textbf{2.63 (-22\%)} & \textbf{1 min (-83\%)} \\
\hline
\end{tabular}
\caption{Ablation study on visual feature representations.}
\label{tab:feature_ablation}
\end{table}

% Table 4: Performance by Instruction Type
\begin{table}[t]
\centering
\begin{tabular}{lcc}
\hline
\textbf{Instruction Type} & \textbf{Samples} & \textbf{SR (\%)} \\
\hline
Turn Left & 104 & 58.7 \\
Turn Right & 96 & 65.6 \\
Go Straight & 109 & 68.8 \\
\hline
\end{tabular}
\caption{Performance breakdown by instruction type.}
\label{tab:instruction_type}
\end{table}

% Table 5: Training Configuration
\begin{table}[t]
\centering
\begin{tabular}{lc}
\hline
\textbf{Hyperparameter} & \textbf{Value} \\
\hline
Batch Size & 16 (effective 32) \\
Learning Rate & 1e-4 (warmup + cosine) \\
Warmup Epochs & 3 \\
Max Epochs & 20 \\
Early Stop Patience & 5 \\
Vocabulary Size & 97 chars \\
Hidden Dimension & 256 \\
Attention Heads & 8 \\
Encoder Layers & 2 \\
\hline
\end{tabular}
\caption{Training hyperparameters.}
\label{tab:hyperparams}
\end{table}
"""

    table_file = OUTPUT_DIR / 'paper_tables.tex'
    with open(table_file, 'w') as f:
        f.write(tables)
    print(f"Saved: {table_file}")


def main():
    """生成所有图表"""
    print("=" * 60)
    print("生成论文图表")
    print("=" * 60)

    print("\n生成训练曲线对比图...")
    plot_training_loss_comparison()

    print("\n生成 SR/SPL 对比图...")
    plot_sr_spl_comparison()

    print("\n生成指令质量对比图...")
    plot_instruction_quality()

    print("\n生成消融实验图...")
    plot_ablation_study()

    print("\n生成架构图...")
    generate_architecture_diagram()

    print("\n生成置信度分布图...")
    plot_confidence_distribution()

    print("\n生成论文表格...")
    generate_all_tables()

    print("\n" + "=" * 60)
    print("所有图表已生成!")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 60)

    # 列出所有生成的文件
    print("\n生成的文件:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
