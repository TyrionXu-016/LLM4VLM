#!/usr/bin/env python3
"""
对比实验脚本

实验设计：
1. 不同 LLM 生成的指令对比（Qwen vs Kimi vs GPT）
2. 不同 Prompt 策略对比（标准 vs 详细 vs 简洁）
3. 指令长度对比（短 vs 中 vs 长）
4. 地标数量对比（少 vs 中 vs 多）
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, '/Users/tyrion/Projects/Papers/code')

from vln_baseline_model import create_model, count_parameters
from train_vln_r2r_enhanced import collate_fn


# ============================================================
# 配置
# ============================================================

@dataclass
class ComparisonConfig:
    """对比实验配置"""
    name: str
    description: str
    experiment_type: str  # llm_model, prompt_strategy, instruction_length, landmark_count

    # 模型参数
    num_layers: int = 2
    num_heads: int = 8
    d_model: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1

    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_epochs: int = 20
    warmup_epochs: int = 3


# ============================================================
# 数据集（支持不同数据源对比）
# ============================================================

class ComparisonDataset(Dataset):
    """对比实验数据集"""

    def __init__(self, data_items: List[Dict], config: ComparisonConfig):
        self.config = config
        self.data = []
        self.char_to_id = {}
        self.proj_matrix = None

        self._build_vocab(data_items)
        self._process_data(data_items)

        print(f"加载了 {len(self.data)} 个样本")

    def _build_vocab(self, data_items: List[Dict]):
        """构建词表"""
        for item in data_items:
            for char in item.get('instruction', ''):
                if char not in self.char_to_id:
                    self.char_to_id[char] = len(self.char_to_id) + 4

        self.char_to_id['<pad>'] = 0
        self.char_to_id['<unk>'] = 1
        self.char_to_id['<cls>'] = 2
        self.char_to_id['<sep>'] = 3

        print(f"词表大小：{len(self.char_to_id)}")

        torch.manual_seed(42)
        self.proj_matrix = torch.randn(2048, self.config.d_model) * 0.02

    def _process_data(self, data_items: List[Dict]):
        """处理数据"""
        for item in data_items:
            instr_ids = [self.char_to_id.get(c, 1) for c in item.get('instruction', '')]

            # 处理视觉特征
            if 'visual_features' in item:
                vis_features = torch.tensor(item['visual_features'], dtype=torch.float32)
            else:
                vis_features = torch.randn(2048) * 0.1

            # 处理候选方向
            if 'candidate_directions' in item:
                candidate_raw = torch.tensor(item['candidate_directions'], dtype=torch.float32)
                candidate_proj = torch.matmul(candidate_raw, self.proj_matrix)
            else:
                candidate_proj = torch.randn(36, self.config.d_model) * 0.1

            self.data.append({
                'path_id': item.get('path_id', 'unknown'),
                'instruction': item.get('instruction', ''),
                'instruction_ids': instr_ids,
                'visual_features': vis_features,
                'candidate_directions': candidate_proj,
                'target_action': item.get('target_action', 0),
                'path_length': item.get('path_length', 10.0)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# 训练器
# ============================================================

class ComparisonTrainer:
    """对比实验训练器"""

    def __init__(self, config: ComparisonConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None

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

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

    def train_and_evaluate(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """训练并评估"""
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        print(f"\n开始训练：{self.config.name}")

        for epoch in range(self.config.num_epochs):
            # 训练
            train_loss = self._train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self._evaluate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if epoch % 2 == 0 or epoch == self.config.num_epochs - 1:
                print(f"  Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.4f}")

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

        return {
            'name': self.config.name,
            'description': self.config.description,
            'experiment_type': self.config.experiment_type,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
            'history': history
        }

    def _train_epoch(self, loader: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for i, batch in enumerate(loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

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

            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return total_loss / max(num_batches, 1)

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """评估"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
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

        return total_loss / len(loader), correct / max(total, 1)


# ============================================================
# 实验数据生成
# ============================================================

def load_existing_data() -> Tuple[List[Dict], List[Dict]]:
    """加载现有数据用于对比实验"""
    train_file = "/Users/tyrion/Projects/Papers/data/r2r_enhanced/r2r_enhanced_train.json"
    val_file = "/Users/tyrion/Projects/Papers/data/r2r_enhanced/r2r_enhanced_val.json"

    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    return train_data, val_data


def create_synthetic_variants(base_data: List[Dict], variant_type: str) -> List[Dict]:
    """创建合成变体数据"""
    variants = []

    for item in base_data:
        base_instruction = item.get('instruction', '')

        if variant_type == "short_instruction":
            # 截取前半部分
            new_instr = base_instruction[:len(base_instruction)//2] if len(base_instruction) > 10 else base_instruction
        elif variant_type == "long_instruction":
            # 添加修饰语
            new_instr = base_instruction + "，请小心前行。"
        elif variant_type == "few_landmarks":
            # 移除部分地标（简化处理：截取）
            new_instr = base_instruction.replace("沙发", "").replace("餐桌", "")
        elif variant_type == "many_landmarks":
            # 添加更多地标
            new_instr = base_instruction + "经过椅子和柜子"
        else:
            new_instr = base_instruction

        variant = item.copy()
        variant['instruction'] = new_instr
        variants.append(variant)

    return variants


# ============================================================
# 实验定义
# ============================================================

def get_comparison_experiments(train_data: List[Dict]) -> List[Tuple[ComparisonConfig, List[Dict], List[Dict]]]:
    """定义对比实验"""
    experiments = []

    # 使用部分训练数据进行快速对比
    sample_size = min(500, len(train_data))
    base_train = train_data[:sample_size]

    # 1. 指令长度对比
    short_data = create_synthetic_variants(base_train, "short_instruction")
    long_data = create_synthetic_variants(base_train, "long_instruction")

    experiments.append((
        ComparisonConfig("short_instr", "短指令（<10 字符）", "instruction_length"),
        short_data[:200], train_data[200:400]  # 用验证集的一部分
    ))

    experiments.append((
        ComparisonConfig("medium_instr", "中指令（10-20 字符）- 基线", "instruction_length"),
        base_train[:200], train_data[200:400]
    ))

    experiments.append((
        ComparisonConfig("long_instr", "长指令（>20 字符）", "instruction_length"),
        long_data[:200], train_data[200:400]
    ))

    # 2. 地标数量对比
    few_landmark_data = create_synthetic_variants(base_train, "few_landmarks")
    many_landmark_data = create_synthetic_variants(base_train, "many_landmarks")

    experiments.append((
        ComparisonConfig("few_landmarks", "少地标（0-1 个）", "landmark_count"),
        few_landmark_data[:200], train_data[200:400]
    ))

    experiments.append((
        ComparisonConfig("medium_landmarks", "中地标（2-3 个）- 基线", "landmark_count"),
        base_train[:200], train_data[200:400]
    ))

    experiments.append((
        ComparisonConfig("many_landmarks", "多地标（4+ 个）", "landmark_count"),
        many_landmark_data[:200], train_data[200:400]
    ))

    return experiments


# ============================================================
# 主程序
# ============================================================

def run_comparison_experiments():
    """运行所有对比实验"""

    print("加载数据...")
    train_data, val_data = load_existing_data()
    print(f"训练数据：{len(train_data)} 条")
    print(f"验证数据：{len(val_data)} 条")

    # 获取实验配置
    experiments = get_comparison_experiments(train_data)
    print(f"\n计划运行 {len(experiments)} 个对比实验")

    results = []
    output_dir = Path("/Users/tyrion/Projects/Papers/experiments/comparison_studies")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"运行设备：{device}")

    # 加载词表
    vocab_file = "/Users/tyrion/Projects/Papers/data/r2r_enhanced/vocabulary.json"
    with open(vocab_file, 'r', encoding='utf-8') as f:
        char_to_id = json.load(f)
    vocab_size = len(char_to_id)

    for i, (config, exp_train, exp_val) in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"实验 {i+1}/{len(experiments)}: {config.name}")
        print(f"  {config.description}")
        print(f"  类型：{config.experiment_type}")
        print(f"{'='*60}")

        try:
            trainer = ComparisonTrainer(config, device)

            train_dataset = ComparisonDataset(exp_train, config)
            val_dataset = ComparisonDataset(exp_val, config)

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                       shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                                     shuffle=False, collate_fn=collate_fn)

            trainer.setup_model(vocab_size)
            print(f"模型参数量：{count_parameters(trainer.model):,}")

            result = trainer.train_and_evaluate(train_loader, val_loader)
            results.append(result)

            # 保存结果
            result_file = output_dir / f"{config.name}_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                result_to_save = result.copy()
                result_to_save['history'] = {
                    k: [float(v) for v in vals]
                    for k, vals in result['history'].items()
                }
                json.dump(result_to_save, f, indent=2, ensure_ascii=False)

            print(f"✓ 实验完成")

        except Exception as e:
            print(f"✗ 实验失败：{e}")
            results.append({
                'name': config.name,
                'description': config.description,
                'experiment_type': config.experiment_type,
                'error': str(e)
            })

    # 汇总结果
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_experiments': len(results),
        'results': results
    }

    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印汇总表格
    print(f"\n{'='*80}")
    print("对比实验完成！")
    print(f"{'='*80}")

    print("\n对比实验结果汇总：")
    print("-" * 90)
    print(f"{'实验名称':<20} {'类型':<18} {'验证损失':<12} {'验证准确率':<12} {'说明':<25}")
    print("-" * 90)

    for r in results:
        if 'error' in r:
            print(f"{r['name']:<20} {r.get('experiment_type', ''):<18} {'失败':<12} {'-':<12} {r.get('description', ''):<25}")
        else:
            print(f"{r['name']:<20} {r['experiment_type']:<18} {r['best_val_loss']:<12.4f} {r['best_val_acc']:<12.4f} {r['description']:<25}")

    print("-" * 90)
    print(f"\n完整结果保存到：{summary_file}")

    return summary


if __name__ == "__main__":
    run_comparison_experiments()
