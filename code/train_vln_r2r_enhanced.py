"""
使用 R2R 增强数据训练 VLN 模型

数据源：data/r2r_enhanced/r2r_enhanced_train.json
特征：ResNet-50 预训练特征（2048 维）
"""

import os
import sys
import json
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../LLM4VLM
DATA_DIR = REPO_ROOT / "data" / "r2r_enhanced"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from vln_baseline_model import VLNBaseline, create_model, count_parameters


# ============================================================
# 数据集 (R2R 增强版)
# ============================================================

@dataclass
class VLNExample:
    """VLN 训练样本"""
    path_id: str
    instruction: str
    instruction_ids: List[int]
    visual_features: List[float]
    candidate_directions: List[List[float]]
    target_action: int
    path_length: float  # 路径长度（用于 SPL 计算）


class R2REnhancedDataset(Dataset):
    """R2R 增强数据集"""

    def __init__(
        self,
        data_file: str,
        feature_dim: int = 2048,
        d_model: int = 256,
        vocab_file: Optional[str] = None,
    ):
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.data = []
        self.char_to_id = {}
        self.vocab_file = vocab_file

        # 学习投影矩阵（2048 -> 256）
        self.proj_matrix = None

        # 加载数据
        self._load_data(data_file)

        print(f"加载了 {len(self.data)} 个样本")

    def _load_data(self, data_file: str):
        """加载 R2R 增强数据"""
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 词表：优先使用生成器输出的 vocabulary.json，确保 train/eval token 映射一致
        if self.vocab_file is not None and os.path.exists(self.vocab_file):
            with open(self.vocab_file, 'r', encoding='utf-8') as vf:
                self.char_to_id = json.load(vf)
        else:
            # 第一遍：根据训练数据构建词表（回退策略）
            for item in raw_data:
                instruction = item['instruction']
                for char in instruction:
                    if char not in self.char_to_id:
                        self.char_to_id[char] = len(self.char_to_id) + 4

            # 添加特殊 token
            self.char_to_id['<pad>'] = 0
            self.char_to_id['<unk>'] = 1
            self.char_to_id['<cls>'] = 2
            self.char_to_id['<sep>'] = 3

        print(f"词表大小：{len(self.char_to_id)}")

        # 学习投影矩阵（使用 PCA 风格的随机投影）
        # 这里使用简单的线性投影
        torch.manual_seed(42)
        self.proj_matrix = torch.randn(self.feature_dim, self.d_model) * 0.02

        # 第二遍：加载数据并投影特征
        for item in raw_data:
            instruction = item.get('instruction', '')
            # 指令 token：直接使用数据文件里的 instruction_ids（避免 train/eval token 不一致）
            instr_ids = item.get('instruction_ids')
            if instr_ids is None:
                instr_ids = [self.char_to_id.get(c, 1) for c in instruction]

            # 投影候选方向 (2048 -> 256)
            candidate_dirs_raw = torch.tensor(item['candidate_directions'], dtype=torch.float32)
            candidate_dirs_proj = torch.matmul(candidate_dirs_raw, self.proj_matrix)

            self.data.append(VLNExample(
                path_id=item['path_id'],
                instruction=instruction,
                instruction_ids=instr_ids,
                visual_features=item['visual_features'],  # 保持 2048 维，模型会处理
                candidate_directions=candidate_dirs_proj.tolist(),  # 投影后 256 维
                target_action=item['target_action'],
                path_length=item.get('path_length', 0.0)
            ))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> VLNExample:
        return self.data[idx]

    def get_vocab_size(self) -> int:
        return len(self.char_to_id)


def collate_fn(batch: List[VLNExample]) -> Dict[str, torch.Tensor]:
    """批处理"""
    instructions = []
    instruction_masks = []
    visual_features = []
    candidate_directions = []
    target_actions = []
    path_lengths = []

    for item in batch:
        # 指令
        instr_ids = torch.tensor([2] + item.instruction_ids + [3], dtype=torch.long)
        instructions.append(instr_ids)

        # mask
        mask = torch.zeros(len(instr_ids), dtype=torch.bool)
        instruction_masks.append(mask)

        # 视觉特征
        visual = torch.tensor(item.visual_features, dtype=torch.float32)
        visual = visual.view(-1, 2048)  # [num_views, 2048]
        visual_features.append(visual)

        # 候选方向
        cand = torch.tensor(item.candidate_directions, dtype=torch.float32)
        candidate_directions.append(cand)

        # 目标动作
        target_actions.append(torch.tensor(item.target_action, dtype=torch.long))

        # 路径长度
        path_lengths.append(item.path_length)

    # padding
    instructions_padded = pad_sequence(instructions, batch_first=True, padding_value=0)
    instruction_masks_padded = pad_sequence(instruction_masks, batch_first=True, padding_value=True)
    visual_features_padded = pad_sequence(visual_features, batch_first=True, padding_value=0)
    candidate_directions_padded = pad_sequence(candidate_directions, batch_first=True, padding_value=0)
    target_actions_tensor = torch.stack(target_actions)
    path_lengths_tensor = torch.tensor(path_lengths, dtype=torch.float32)

    return {
        'instructions': instructions_padded,
        'instruction_mask': instruction_masks_padded,
        'visual_features': visual_features_padded,
        'candidate_directions': candidate_directions_padded,
        'target_action': target_actions_tensor,
        'path_lengths': path_lengths_tensor
    }


# ============================================================
# 训练器 (R2R 增强版)
# ============================================================

class R2REnhancedTrainer:
    """R2R 增强训练器"""

    def __init__(self, model: VLNBaseline, train_loader: DataLoader,
                 val_loader: DataLoader = None, lr: float = 1e-4,
                 weight_decay: float = 1e-4, device: str = 'cpu',
                 grad_accum_steps: int = 4, warmup_epochs: int = 3,
                 patience: int = 5):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=20,
            eta_min=1e-6
        )

        # 早停
        self.patience = patience
        self.epochs_without_improvement = 0

        self.criterion = nn.CrossEntropyLoss()

        self.best_val_loss = float('inf')
        self.train_history = []
        self.grad_accum_steps = grad_accum_steps
        self.warmup_epochs = warmup_epochs

    def _warmup_lr(self, epoch: int):
        """学习率预热"""
        if epoch <= self.warmup_epochs:
            factor = epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 1e-4 * factor

    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            output = self.model(batch)
            loss = output['loss'] / self.grad_accum_steps

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 梯度累积
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

            # 打印进度
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item() * self.grad_accum_steps:.4f}, LR: {current_lr:.6f}")

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def validate(self, epoch: int) -> float:
        """验证"""
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(batch)
                total_loss += output['loss'].item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Validation Epoch {epoch}, Loss: {avg_loss:.4f}")

        # 早停检查
        if avg_loss < self.best_val_loss - 0.001:
            self.best_val_loss = avg_loss
            self.epochs_without_improvement = 0
            self.save_checkpoint('best')
            print(f"  ✓ 保存最佳模型 (损失：{avg_loss:.4f})")
        else:
            self.epochs_without_improvement += 1
            print(f"  无改进轮数：{self.epochs_without_improvement}/{self.patience}")

        return avg_loss

    def save_checkpoint(self, name: str = 'checkpoint'):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history
        }
        path = CHECKPOINT_DIR / f"vln_r2r_{name}.pt"
        os.makedirs(path.parent, exist_ok=True)
        torch.save(checkpoint, str(path))
        print(f"  已保存模型：{path}")

    def train(self, num_epochs: int = 20):
        """训练"""
        print("=" * 70)
        print("R2R 增强数据 VLN 训练 (ResNet 特征)")
        print("=" * 70)
        print(f"设备：{self.device}")
        print(f"批次大小：{self.train_loader.batch_size}")
        print(f"梯度累积：{self.grad_accum_steps}")
        print(f"训练批次数：{len(self.train_loader)}")
        print(f"模型参数：{count_parameters(self.model):,}")
        print(f"学习率：{self.optimizer.param_groups[0]['lr']}")
        print(f"早停耐心值：{self.patience}")
        print(f"预热轮数：{self.warmup_epochs}")
        print()

        start_time = datetime.now()

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)

            # 预热
            self._warmup_lr(epoch)

            train_loss = self.train_epoch(epoch)
            print(f"  Training Loss: {train_loss:.4f}")

            val_loss = self.validate(epoch)

            # 学习率调度 (预热后)
            if epoch > self.warmup_epochs:
                self.scheduler.step()

            self.train_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr'],
                'best_val_loss': self.best_val_loss
            })

            # 早停
            if self.epochs_without_improvement >= self.patience:
                print(f"\n早停触发：{self.patience} 轮无改进")
                break

            # 保存历史
            self._save_history()

        elapsed = datetime.now() - start_time
        print("\n" + "=" * 70)
        print("训练完成!")
        print(f"耗时：{elapsed}")
        print(f"最佳验证损失：{self.best_val_loss:.4f}")
        print(f"完成轮数：{len(self.train_history)}/{num_epochs}")
        print("=" * 70)

        self.save_checkpoint('final')
        self._save_history()

    def _save_history(self):
        """保存训练历史"""
        history_file = CHECKPOINT_DIR / "training_history_r2r.json"
        os.makedirs(history_file.parent, exist_ok=True)
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-data", type=str, default=str(DATA_DIR / "r2r_enhanced_train.json"))
    parser.add_argument("--val-data", type=str, default=str(DATA_DIR / "r2r_enhanced_val.json"))
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.lr
    grad_accum_steps = args.grad_accum_steps  # 有效 batch_size = batch_size * grad_accum_steps
    patience = args.patience
    warmup_epochs = args.warmup_epochs
    device = args.device

    train_data_file = args.train_data
    val_data_file = args.val_data
    vocab_file = str(DATA_DIR / "vocabulary.json")

    print(f"使用设备：{device}")
    print(f"有效批次大小：{batch_size * grad_accum_steps}")
    print(f"训练数据：{train_data_file}")
    print(f"验证数据：{val_data_file}")

    # 创建数据集
    print("\n创建数据集...")
    train_dataset = R2REnhancedDataset(data_file=train_data_file, vocab_file=vocab_file)
    val_dataset = R2REnhancedDataset(data_file=val_data_file, vocab_file=vocab_file)

    vocab_size = train_dataset.get_vocab_size()
    print(f"词表大小：{vocab_size}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 创建模型
    print("\n创建模型...")
    model = create_model(vocab_size=vocab_size, d_model=256)
    print(f"模型参数：{count_parameters(model):,}")

    # 创建训练器
    trainer = R2REnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        device=device,
        grad_accum_steps=grad_accum_steps,
        warmup_epochs=warmup_epochs,
        patience=patience
    )

    # 开始训练
    trainer.train(num_epochs=num_epochs)

    print("\n训练完成!")


if __name__ == "__main__":
    main()
