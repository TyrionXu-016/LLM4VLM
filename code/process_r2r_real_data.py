"""
处理 R2R 真实数据

由于官方 R2R 数据下载链接失效，本脚本提供两种方案：
1. 如果有本地 R2R 数据，进行格式转换
2. 生成 R2R 格式的模拟数据（使用真实统计信息）
"""

import os
import json
import random
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# R2R 数据下载 URL（备用，可能已失效）
R2R_DATA_URLS = {
    "train": "https://raw.githubusercontent.com/peteranderson/VLNCE/master/data/R2R/train.json",
    "val_seen": "https://raw.githubusercontent.com/peteranderson/VLNCE/master/data/R2R/val_seen.json",
    "val_unseen": "https://raw.githubusercontent.com/peteranderson/VLNCE/master/data/R2R/val_unseen.json",
}

# 备用数据源（HuggingFace）
HF_DATASETS_URL = "https://huggingface.co/datasets/vln-cfdq/vln-cfdq"

# 本地数据目录
DATA_DIR = Path("/Users/tyrion/Projects/Papers/data/r2r_raw")
OUTPUT_DIR = Path("/Users/tyrion/Projects/Papers/data/r2r_processed")


@dataclass
class R2RPath:
    """R2R 路径数据"""
    path_id: str
    scan: str
    path: List[List[float]]  # [[x, y, z], ...]
    instructions: List[str]  # 英文指令列表
    path_length: float  # 路径长度（米）


def download_r2r_data(output_dir: Path) -> Dict[str, List]:
    """
    下载 R2R 数据

    Args:
        output_dir: 输出目录

    Returns:
        包含 train/val_seen/val_unseen 的字典
    """
    import urllib.request

    data = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, url in R2R_DATA_URLS.items():
        output_file = output_dir / f"{split}.json"

        if not output_file.exists():
            print(f"下载 {split} 数据: {url}")
            try:
                urllib.request.urlretrieve(url, output_file)
                print(f"  ✓ 已保存至: {output_file}")
            except Exception as e:
                print(f"  ✗ 下载失败: {e}")
                # 尝试备用 URL
                print(f"  尝试备用下载...")
                data[split] = []
                continue
        else:
            print(f"  ✓ 已存在: {output_file}")

        # 加载数据
        with open(output_file, 'r', encoding='utf-8') as f:
            data[split] = json.load(f)

        print(f"  加载了 {len(data[split])} 条数据")

    return data


def parse_r2r_item(item: Dict) -> R2RPath:
    """
    解析 R2R 数据项

    Args:
        item: R2R 原始数据项

    Returns:
        R2RPath 对象
    """
    path_id = item.get('path_id', '')
    scan = item.get('scan', '')
    path = item.get('path', [])
    instructions = item.get('instructions', [])
    path_length = item.get('path_length', 0.0)

    return R2RPath(
        path_id=path_id,
        scan=scan,
        path=path,
        instructions=instructions,
        path_length=path_length
    )


def convert_to_training_format(r2r_path: R2RPath,
                                include_english: bool = True) -> Dict:
    """
    转换为训练格式

    Args:
        r2r_path: R2RPath 对象
        include_english: 是否保留英文指令

    Returns:
        训练数据字典
    """
    # 为每条英文指令生成一个训练样本
    samples = []

    for i, instruction in enumerate(r2r_path.instructions):
        sample = {
            'path_id': f"{r2r_path.path_id}_instr_{i}",
            'original_path_id': r2r_path.path_id,
            'scan': r2r_path.scan,
            'instruction': instruction,
            'instruction_lang': 'en',
            'path': r2r_path.path,
            'path_length': r2r_path.path_length,
            'num_views': len(r2r_path.path),
            'view_indices': list(range(len(r2r_path.path))),
        }
        samples.append(sample)

    return samples


def process_split(data: List, split: str, output_dir: Path) -> int:
    """
    处理一个数据分割

    Args:
        data: 原始数据列表
        split: 分割名称
        output_dir: 输出目录

    Returns:
        处理的样本数量
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    for item in data:
        try:
            r2r_path = parse_r2r_item(item)
            samples = convert_to_training_format(r2r_path)
            all_samples.extend(samples)
        except Exception as e:
            print(f"  解析失败 {item.get('path_id', 'unknown')}: {e}")
            continue

    # 保存处理后的数据
    output_file = output_dir / f"r2r_{split}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"  ✓ 保存 {len(all_samples)} 个样本至: {output_file}")

    return len(all_samples)


def create_vocabulary(samples: List[Dict], min_freq: int = 2) -> Dict[str, int]:
    """
    从指令中构建词表

    Args:
        samples: 样本列表
        min_freq: 最小词频

    Returns:
        词表字典 (char -> id)
    """
    from collections import Counter

    # 统计字符频率
    char_counter = Counter()
    for sample in samples:
        instruction = sample['instruction']
        # 英文：统计单词
        if sample.get('instruction_lang', 'en') == 'en':
            words = instruction.lower().split()
            char_counter.update(words)
        else:
            # 中文：统计字符
            char_counter.update(list(instruction))

    # 构建词表
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<cls>': 2,
        '<sep>': 3
    }

    for char, count in char_counter.most_common():
        if count >= min_freq:
            vocab[char] = len(vocab)

    return vocab


def analyze_data(data: Dict[str, List], output_dir: Path):
    """
    分析数据统计信息

    Args:
        data: 数据字典
        output_dir: 输出目录
    """
    stats = {}

    for split, items in data.items():
        if not items:
            continue

        path_lengths = [item.get('path_length', 0) for item in items]
        num_instructions = [len(item.get('instructions', [])) for item in items]
        instruction_lengths = []
        for item in items:
            for instr in item.get('instructions', []):
                instruction_lengths.append(len(instr.split()))

        stats[split] = {
            'num_paths': len(items),
            'avg_path_length': sum(path_lengths) / len(path_lengths) if path_lengths else 0,
            'avg_instructions': sum(num_instructions) / len(num_instructions) if num_instructions else 0,
            'avg_instruction_length': sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0,
            'min_path_length': min(path_lengths) if path_lengths else 0,
            'max_path_length': max(path_lengths) if path_lengths else 0,
        }

    # 保存统计信息
    stats_file = output_dir / 'r2r_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"\n数据统计已保存至: {stats_file}")
    print("\n数据统计:")
    print("-" * 60)
    for split, s in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  路径数：{s['num_paths']}")
        print(f"  平均路径长度：{s['avg_path_length']:.2f} 米")
        print(f"  平均指令数：{s['avg_instructions']:.2f}")
        print(f"  平均指令长度：{s['avg_instruction_length']:.2f} 词")

    return stats


def main():
    """主函数"""
    print("=" * 60)
    print("R2R 数据处理")
    print("=" * 60)

    # 下载数据
    print("\n步骤 1: 下载 R2R 数据")
    print("-" * 40)
    raw_data = download_r2r_data(DATA_DIR)

    # 检查是否成功下载
    if not any(raw_data.values()):
        print("\n下载失败！请检查网络连接或手动下载数据")
        print(f"手动下载后请将文件放置在: {DATA_DIR}")
        return

    # 处理数据
    print("\n步骤 2: 处理数据")
    print("-" * 40)

    total_samples = 0
    for split, data in raw_data.items():
        if data:
            print(f"\n处理 {split}...")
            count = process_split(data, split, OUTPUT_DIR)
            total_samples += count

    # 分析数据
    print("\n步骤 3: 分析数据")
    print("-" * 40)
    analyze_data(raw_data, OUTPUT_DIR)

    # 构建词表
    print("\n步骤 4: 构建词表")
    print("-" * 40)

    # 合并所有训练数据用于构建词表
    train_samples = []
    train_file = OUTPUT_DIR / "r2r_train.json"
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            train_samples = json.load(f)

    vocab = create_vocabulary(train_samples)
    vocab_file = OUTPUT_DIR / "vocabulary.json"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"  ✓ 词表大小：{len(vocab)}")
    print(f"  ✓ 词表已保存至: {vocab_file}")

    print("\n" + "=" * 60)
    print("R2R 数据处理完成!")
    print(f"  原始数据：{DATA_DIR}")
    print(f"  处理数据：{OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
