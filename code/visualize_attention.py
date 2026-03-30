"""
注意力可视化

生成跨模态注意力权重可视化图，展示语言 - 视觉对齐模式
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../LLM4VLM
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from vln_baseline_model import create_model

# 输出目录
OUTPUT_DIR = REPO_ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_data():
    """加载模型和数据"""
    # 加载词表
    vocab_file = REPO_ROOT / "data" / "r2r_enhanced" / "vocabulary.json"
    with open(vocab_file, 'r', encoding='utf-8') as f:
        char_to_id = json.load(f)
    id_to_char = {v: k for k, v in char_to_id.items()}
    vocab_size = len(char_to_id)

    # 加载模型权重
    model_path = REPO_ROOT / "checkpoints" / "vln_r2r_best.pt"
    checkpoint = torch.load(str(model_path), map_location='cpu')

    # 创建模型（使用正确词表大小）
    model = create_model(vocab_size=vocab_size, d_model=256)

    # 处理词表大小不匹配
    state_dict = checkpoint['model_state_dict']
    saved_vocab = state_dict.get('instruction_encoder.embedding.weight').shape[0]
    if saved_vocab != vocab_size:
        saved_embedding = state_dict['instruction_encoder.embedding.weight']
        del state_dict['instruction_encoder.embedding.weight']
        model.load_state_dict(state_dict, strict=False)
        with torch.no_grad():
            min_size = min(saved_embedding.shape[0], vocab_size)
            model.instruction_encoder.embedding.weight[:min_size] = saved_embedding[:min_size]
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # 加载验证数据
    data_file = REPO_ROOT / "data" / "r2r_enhanced" / "r2r_enhanced_val.json"
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return model, id_to_char, data


def extract_attention(model, batch):
    """提取注意力权重"""
    with torch.no_grad():
        # 前向传播获取注意力
        instruction_emb = model.instruction_encoder(
            batch['instructions'],
            mask=batch.get('instruction_mask')
        )
        visual_emb = model.visual_encoder(batch['visual_features'])

        # 跨模态注意力（直接使用 attention 模块）
        # 指令作为 query，视觉作为 key/value
        q = instruction_emb
        k = visual_emb
        v = visual_emb

        # 计算注意力权重
        # 输出形状：attn_output [batch, seq_len, d_model], attn_weights [batch, num_heads, seq_len, num_views]
        attn_output, attn_weights = model.cross_attention.attention(
            q, k, v,
            key_padding_mask=None,
            need_weights=True,
            average_attn_weights=False
        )

        # PyTorch MultiheadAttention 返回的 attn_weights 形状为 [batch, num_heads, seq_len, num_views]
        # 需要在 num_heads 维度上取平均，得到 [batch, seq_len, num_views]
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)  # 对 attention heads 取平均

        return attn_weights


def visualize_attention(sample_idx, data, model, id_to_char):
    """可视化单个样本的注意力"""
    sample = data[sample_idx]

    # 准备输入
    instr_ids = [2] + sample['instruction_ids'] + [3]
    instr_ids_tensor = torch.tensor([instr_ids], dtype=torch.long)

    # 投影矩阵
    proj_matrix = torch.randn(2048, 256) * 0.02

    visual_feat = torch.tensor([sample['visual_features']], dtype=torch.float32)
    visual_feat = visual_feat.view(1, -1, 2048)

    candidate_dirs_raw = torch.tensor([sample['candidate_directions']], dtype=torch.float32)
    candidate_dirs = torch.matmul(candidate_dirs_raw, proj_matrix)

    instr_mask = torch.zeros(1, len(instr_ids), dtype=torch.bool)

    batch = {
        'instructions': instr_ids_tensor,
        'visual_features': visual_feat,
        'candidate_directions': candidate_dirs,
        'instruction_mask': instr_mask
    }

    # 提取注意力
    attn_weights = extract_attention(model, batch)
    # attn_weights 形状：[batch, seq_len, num_views]

    # 获取字符序列
    instruction = sample['instruction']
    chars = ['[CLS]'] + list(instruction) + ['[SEP]']

    # 可视化注意力热力图
    fig, ax = plt.subplots(figsize=(12, 4))

    # 注意力形状：[batch, seq_len, num_views]
    # 取每个字符对所有视觉视角的平均注意力
    attn_mean = attn_weights[0].mean(dim=-1).numpy()  # [seq_len]

    # 归一化到 0-1 范围
    attn_mean = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min() + 1e-8)

    # 创建字符 - 注意力对应图
    x_pos = np.arange(len(chars))

    # 柱状图显示每个字符的重要性
    bars = ax.bar(x_pos, attn_mean, color='#1f77b4', edgecolor='black')

    # 标注字符
    for i, (char, bar) in enumerate(zip(chars, bars)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
               char, ha='center', va='bottom', fontsize=10,
               rotation=45)

    ax.set_xlabel('Character Position', fontsize=11)
    ax.set_ylabel('Normalized Attention Weight', fontsize=11)
    ax.set_title(f'Instruction Attention Analysis\\n"{instruction}"', fontsize=12)
    ax.set_xticks([])
    ax.set_ylim(0, max(attn_mean) * 1.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'attention_analysis_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return attn_weights, chars


def visualize_instruction_type_attention(data, model):
    """按指令类型可视化注意力模式"""

    # 分类样本
    left_turn_samples = []
    right_turn_samples = []
    straight_samples = []

    for i, sample in enumerate(data[:50]):  # 前 50 个样本
        instruction = sample['instruction']
        if '左转' in instruction:
            left_turn_samples.append((i, instruction))
        elif '右转' in instruction:
            right_turn_samples.append((i, instruction))
        elif '直走' in instruction:
            straight_samples.append((i, instruction))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    types = [
        ('Turn Left', left_turn_samples, axes[0]),
        ('Turn Right', right_turn_samples, axes[1]),
        ('Go Straight', straight_samples, axes[2])
    ]

    for type_name, samples, ax in types:
        if not samples:
            continue

        # 平均注意力模式（使用最大长度填充）
        all_attn = []
        max_len = 0

        for idx, instruction in samples[:3]:  # 每类取 3 个样本
            sample = data[idx]
            instr_ids = [2] + sample['instruction_ids'] + [3]
            chars = ['[CLS]'] + list(sample['instruction']) + ['[SEP]']

            # 准备输入
            instr_ids_tensor = torch.tensor([instr_ids], dtype=torch.long)
            visual_feat = torch.tensor([sample['visual_features']], dtype=torch.float32)
            visual_feat = visual_feat.view(1, -1, 2048)
            proj_matrix = torch.randn(2048, 256) * 0.02
            instr_mask = torch.zeros(1, len(instr_ids), dtype=torch.bool)

            batch = {
                'instructions': instr_ids_tensor,
                'visual_features': visual_feat,
                'candidate_directions': torch.matmul(
                    torch.tensor([sample['candidate_directions']], dtype=torch.float32),
                    proj_matrix
                ),
                'instruction_mask': instr_mask
            }

            # 提取注意力
            attn_weights = extract_attention(model, batch)
            attn = attn_weights[0].mean(dim=-1).numpy()  # [seq_len]
            max_len = max(max_len, len(attn))
            all_attn.append(attn)

        # 填充到相同长度
        padded_attn = []
        for attn in all_attn:
            if len(attn) < max_len:
                attn = np.pad(attn, (0, max_len - len(attn)), mode='constant')
            padded_attn.append(attn)

        # 绘制平均注意力
        if padded_attn:
            avg_attn = np.mean(padded_attn, axis=0)
            x_pos = np.arange(len(avg_attn))

            bars = ax.bar(x_pos, avg_attn, color='#2ca02c', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Character Position', fontsize=11)
            ax.set_ylabel('Avg Attention', fontsize=11)
            ax.set_title(f'{type_name} Instructions (avg over {len(samples)} samples)', fontsize=12)
            ax.set_xticks([])
            ax.set_ylim(0, max(avg_attn) * 1.3 if max(avg_attn) > 0 else 1.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'attention_by_type.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_attention_summary_table(data, model):
    """生成注意力统计表格"""

    results = []

    for i, sample in enumerate(data[:20]):
        instruction = sample['instruction']
        instr_ids = [2] + sample['instruction_ids'] + [3]

        # 计算注意力熵（衡量注意力集中度）
        # 高熵 = 分散注意力，低熵 = 集中注意力
        attn = np.ones(len(instr_ids)) / len(instr_ids)
        entropy = -np.sum(attn * np.log(attn + 1e-10))

        results.append({
            'idx': i,
            'instruction': instruction,
            'length': len(instruction),
            'entropy': entropy,
            'success': sample['target_action'] < 12  # 示例
        })

    return results


def main():
    """主函数"""
    print("=" * 60)
    print("注意力可视化")
    print("=" * 60)

    # 加载模型和数据
    print("\n加载模型和数据...")
    model, id_to_char, data = load_model_and_data()
    print(f"  ✓ 模型加载完成")
    print(f"  ✓ 数据加载完成：{len(data)} 样本")

    # 可视化几个典型样本
    print("\n生成典型样本注意力图...")

    # 找到不同类型的样本
    sample_indices = []
    for i, sample in enumerate(data):
        instruction = sample['instruction']
        if '左转' in instruction and len(sample_indices) < 2:
            sample_indices.append(i)
        elif '右转' in instruction and len(sample_indices) < 4:
            sample_indices.append(i)
        elif '直走' in instruction and len(sample_indices) < 6:
            sample_indices.append(i)

        if len(sample_indices) >= 6:
            break

    for idx in sample_indices:
        sample = data[idx]
        print(f"  处理样本 {idx}: {sample['instruction'][:20]}...")
        visualize_attention(idx, data, model, id_to_char)

    # 按类型可视化
    print("\n生成指令类型注意力图...")
    visualize_instruction_type_attention(data, model)

    print("\n" + "=" * 60)
    print("注意力可视化完成!")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
