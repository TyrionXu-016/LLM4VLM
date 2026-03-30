#!/usr/bin/env python3
"""
消融实验脚本

实验设计：
1. 模型架构消融 - 不同 Transformer 层数、注意力头数
2. 视觉特征消融 - 随机特征 vs ResNet vs 不同投影维度
3. 训练策略消融 - 学习率、批量大小、梯度累积
4. 数据量消融 - 不同训练数据规模的影响
5. Prompt 策略消融 - 不同 prompt 设计的影响
"""

import os
import sys
import json
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import math

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../LLM4VLM
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from vln_baseline_model import VLNBaseline, create_model, count_parameters


# ============================================================
# 配置
# ============================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    description: str

    # 模型参数
    num_layers: int = 2
    num_heads: int = 8
    d_model: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1

    # 视觉特征
    visual_feature_dim: int = 2048
    feature_type: str = "resnet"  # resnet, random, projected

    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_epochs: int = 20
    warmup_epochs: int = 3

    # 数据参数
    train_samples: int = 1000  # 用于数据量消融
    val_samples: int = 200

    # Prompt 参数（用于 prompt 消融）
    prompt_type: str = "standard"  # standard, detailed, minimal


# ============================================================
# 数据集
# ============================================================

class R2REnhancedDataset(Dataset):
    """R2R 增强数据集"""

    def __init__(self, data_file: str, config: ExperimentConfig):
        self.config = config
        self.data = []
        self.char_to_id = {}
        self.proj_matrix = None

        self._load_data(data_file)
        print(f"加载了 {len(self.data)} 个样本")

    def _load_data(self, data_file: str):
        """加载数据"""
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 第一遍：构建词表
        for item in raw_data:
            for char in item['instruction']:
                if char not in self.char_to_id:
                    self.char_to_id[char] = len(self.char_to_id) + 4

        # 特殊 token
        self.char_to_id['<pad>'] = 0
        self.char_to_id['<unk>'] = 1
        self.char_to_id['<cls>'] = 2
        self.char_to_id['<sep>'] = 3

        print(f"词表大小：{len(self.char_to_id)}")

        # 投影矩阵
        torch.manual_seed(42)
        if self.config.feature_type == "random":
            self.proj_matrix = torch.randn(self.config.visual_feature_dim, self.config.d_model) * 0.5
        else:
            self.proj_matrix = torch.randn(self.config.visual_feature_dim, self.config.d_model) * 0.02

        # 第二遍：加载数据
        for i, item in enumerate(raw_data):
            if i >= self.config.train_samples and "train" in data_file:
                break

            instr_ids = [self.char_to_id.get(c, 1) for c in item['instruction']]

            candidate_dirs_raw = torch.tensor(item['candidate_directions'], dtype=torch.float32)
            candidate_dirs_proj = torch.matmul(candidate_dirs_raw, self.proj_matrix)

            self.data.append({
                'path_id': item['path_id'],
                'instruction': item['instruction'],
                'instruction_ids': instr_ids,
                'visual_features': torch.tensor(item['visual_features'], dtype=torch.float32),
                'candidate_directions': candidate_dirs_proj,
                'target_action': item['target_action'],
                'path_length': item.get('path_length', 10.0)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """批量数据 collate"""
    instructions = []
    visual_features = []
    candidate_directions = []
    target_actions = []
    path_lengths = []
    path_ids = []
    instr_texts = []

    for item in batch:
        instructions.append(torch.tensor(item['instruction_ids'], dtype=torch.long))
        visual_features.append(item['visual_features'])
        candidate_directions.append(item['candidate_directions'])
        target_actions.append(item['target_action'])
        path_lengths.append(item['path_length'])
        path_ids.append(item['path_id'])
        instr_texts.append(item['instruction'])

    return {
        'instructions': nn.utils.rnn.pad_sequence(instructions, batch_first=True, padding_value=0),
        'visual_features': torch.stack(visual_features),
        'candidate_directions': torch.stack(candidate_directions),
        'target_actions': torch.tensor(target_actions, dtype=torch.long),
        'path_lengths': torch.tensor(path_lengths, dtype=torch.float32),
        'path_ids': path_ids,
        'instr_texts': instr_texts
    }


# ============================================================
# 训练器
# ============================================================

class AblationTrainer:
    """消融实验训练器"""

    def __init__(self, config: ExperimentConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def setup_model(self, vocab_size: int):
        """初始化模型"""
        self.model = create_model(
            vocab_size=vocab_size,
            d_model=self.config.d_model,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        )
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs - self.config.warmup_epochs
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # 前向传播
            outputs = self.model(
                batch['instructions'],
                batch['visual_features'],
                batch['candidate_directions']
            )

            loss = nn.functional.cross_entropy(outputs, batch['target_actions'])
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # 梯度更新
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return total_loss / num_batches

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                outputs = self.model(
                    batch['instructions'],
                    batch['visual_features'],
                    batch['candidate_directions']
                )

                loss = nn.functional.cross_entropy(outputs, batch['target_actions'])
                total_loss += loss.item()

                predictions = outputs.argmax(dim=-1)
                correct += (predictions == batch['target_actions']).sum().item()
                total += batch['target_actions'].size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """完整训练流程"""
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        print(f"\n开始训练：{self.config.name}")
        print(f"  配置：layers={self.config.num_layers}, heads={self.config.num_heads}, "
              f"d_model={self.config.d_model}, lr={self.config.learning_rate}")

        for epoch in range(self.config.num_epochs):
            # Warmup
            if epoch < self.config.warmup_epochs:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs

            # 训练
            train_loss = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.evaluate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"  Epoch {epoch+1}/{self.config.num_epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"  早停于 epoch {epoch+1}")
                    break

            self.scheduler.step()

        return {
            'name': self.config.name,
            'description': self.config.description,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
            'history': history,
            'config': asdict(self.config)
        }


# ============================================================
# 实验定义
# ============================================================

def get_ablation_experiments() -> List[ExperimentConfig]:
    """定义消融实验"""
    experiments = []

    # 1. 基线配置
    experiments.append(ExperimentConfig(
        name="baseline",
        description="基线配置（与论文一致）",
        num_layers=2,
        num_heads=8,
        d_model=256,
        learning_rate=1e-4,
        batch_size=16,
        feature_type="resnet"
    ))

    # 2. 模型架构消融
    experiments.append(ExperimentConfig(
        name="1_layer",
        description="1 层 Transformer（减少模型容量）",
        num_layers=1,
        num_heads=8,
        d_model=256,
        feature_type="resnet"
    ))

    experiments.append(ExperimentConfig(
        name="4_layer",
        description="4 层 Transformer（增加模型容量）",
        num_layers=4,
        num_heads=8,
        d_model=256,
        feature_type="resnet"
    ))

    experiments.append(ExperimentConfig(
        name="4_heads",
        description="4 注意力头（减少注意力头数）",
        num_layers=2,
        num_heads=4,
        d_model=256,
        feature_type="resnet"
    ))

    experiments.append(ExperimentConfig(
        name="16_heads",
        description="16 注意力头（增加注意力头数）",
        num_layers=2,
        num_heads=16,
        d_model=256,
        feature_type="resnet"
    ))

    # 3. 隐藏维度消融
    experiments.append(ExperimentConfig(
        name="d_model_128",
        description="d_model=128（减少隐藏维度）",
        num_layers=2,
        num_heads=8,
        d_model=128,
        feature_type="resnet"
    ))

    experiments.append(ExperimentConfig(
        name="d_model_512",
        description="d_model=512（增加隐藏维度）",
        num_layers=2,
        num_heads=8,
        d_model=512,
        feature_type="resnet"
    ))

    # 4. 学习率消融
    experiments.append(ExperimentConfig(
        name="lr_5e-5",
        description="学习率 5e-5（更保守）",
        num_layers=2,
        num_heads=8,
        d_model=256,
        learning_rate=5e-5,
        feature_type="resnet"
    ))

    experiments.append(ExperimentConfig(
        name="lr_2e-4",
        description="学习率 2e-4（更激进）",
        num_layers=2,
        num_heads=8,
        d_model=256,
        learning_rate=2e-4,
        feature_type="resnet"
    ))

    # 5. 数据量消融
    experiments.append(ExperimentConfig(
        name="data_250",
        description="250 训练样本（25% 数据）",
        num_layers=2,
        num_heads=8,
        d_model=256,
        train_samples=250,
        feature_type="resnet"
    ))

    experiments.append(ExperimentConfig(
        name="data_500",
        description="500 训练样本（50% 数据）",
        num_layers=2,
        num_heads=8,
        d_model=256,
        train_samples=500,
        feature_type="resnet"
    ))

    experiments.append(ExperimentConfig(
        name="data_1500",
        description="1500 训练样本（150% 数据）",
        num_layers=2,
        num_heads=8,
        d_model=256,
        train_samples=1500,
        feature_type="resnet"
    ))

    # 6. 视觉特征消融
    experiments.append(ExperimentConfig(
        name="random_feature",
        description="随机高斯特征（无预训练）",
        num_layers=2,
        num_heads=8,
        d_model=256,
        feature_type="random"
    ))

    return experiments


# ============================================================
# 主程序
# ============================================================

def run_ablation_studies():
    """运行所有消融实验"""

    # 数据文件路径
    train_file = str(REPO_ROOT / "data" / "r2r_enhanced" / "r2r_enhanced_train.json")
    val_file = str(REPO_ROOT / "data" / "r2r_enhanced" / "r2r_enhanced_val.json")

    # 加载词表获取词表大小
    vocab_file = str(REPO_ROOT / "data" / "r2r_enhanced" / "vocabulary.json")
    with open(vocab_file, 'r', encoding='utf-8') as f:
        char_to_id = json.load(f)
    vocab_size = len(char_to_id)

    print(f"词表大小：{vocab_size}")
    print(f"训练数据：{train_file}")
    print(f"验证数据：{val_file}")

    # 获取实验配置
    experiments = get_ablation_experiments()
    print(f"\n计划运行 {len(experiments)} 个消融实验")

    results = []
    output_dir = REPO_ROOT / "experiments" / "ablation_studies"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"运行设备：{device}")

    for i, config in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"实验 {i+1}/{len(experiments)}: {config.name}")
        print(f"  {config.description}")
        print(f"{'='*60}")

        try:
            # 创建训练器和数据加载器
            trainer = AblationTrainer(config, device)

            train_dataset = R2REnhancedDataset(train_file, config)
            val_dataset = R2REnhancedDataset(val_file, config)

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                       shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                                     shuffle=False, collate_fn=collate_fn)

            # 初始化模型
            trainer.setup_model(vocab_size)
            print(f"模型参数量：{count_parameters(trainer.model):,}")

            # 训练
            result = trainer.train(train_loader, val_loader)
            results.append(result)

            # 保存结果
            result_file = output_dir / f"{config.name}_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                # 转换历史中的张量为列表
                result_to_save = result.copy()
                result_to_save['history'] = {
                    k: [float(v) for v in vals]
                    for k, vals in result['history'].items()
                }
                json.dump(result_to_save, f, indent=2, ensure_ascii=False)

            print(f"✓ 实验完成，结果保存到 {result_file}")

        except Exception as e:
            print(f"✗ 实验失败：{e}")
            results.append({
                'name': config.name,
                'description': config.description,
                'error': str(e)
            })

    # 汇总结果
    summary_file = output_dir / "ablation_summary.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_experiments': len(results),
        'results': results
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("消融实验完成！")
    print(f"{'='*60}")

    # 打印汇总表格
    print("\n消融实验结果汇总：")
    print("-" * 80)
    print(f"{'实验名称':<20} {'验证损失':<12} {'验证准确率':<12} {'说明':<30}")
    print("-" * 80)

    for r in results:
        if 'error' in r:
            print(f"{r['name']:<20} {'失败':<12} {'-':<12} {r.get('description', ''):<30}")
        else:
            print(f"{r['name']:<20} {r['best_val_loss']:<12.4f} {r['best_val_acc']:<12.4f} {r['description']:<30}")

    print("-" * 80)
    print(f"\n完整结果保存到：{summary_file}")

    return summary


if __name__ == "__main__":
    run_ablation_studies()
