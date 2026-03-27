"""
VLN 基线模型训练脚本

包含：数据加载、训练循环、评估
"""

import os
import sys
import json
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, '/Users/tyrion/Projects/Papers/code')

from vln_baseline_model import VLNBaseline, create_model, count_parameters


# ============================================================
# 数据集
# ============================================================

@dataclass
class VLNExample:
    """VLN 训练样本"""
    path_id: str
    instruction: str
    instruction_ids: List[int]  # token IDs
    visual_features: List[float]  # 展平的特征
    candidate_directions: List[List[float]]  # 候选方向特征
    target_action: int  # 目标动作索引


class VLNDataset(Dataset):
    """VLN 数据集（使用模拟数据）"""

    def __init__(self, data_file: str = None, vocab_size: int = 5000,
                 num_samples: int = 100, feature_dim: int = 2048):
        """
        Args:
            data_file: 数据文件路径（如无则生成模拟数据）
            vocab_size: 词表大小
            num_samples: 样本数量（模拟数据用）
            feature_dim: 视觉特征维度
        """
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.data = []

        if data_file and os.path.exists(data_file):
            self._load_data(data_file)
        else:
            self._generate_synthetic_data(num_samples)

    def _load_data(self, data_file: str):
        """加载真实数据"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # TODO: 转换为 VLNExample
        print(f"加载了 {len(data)} 条数据")

    def _generate_synthetic_data(self, num_samples: int):
        """生成模拟训练数据"""
        random.seed(42)

        # 简单词表映射（字 -> ID）
        self.char_to_id = {
            '<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3
        }

        # 常见导航用字
        nav_chars = list("直走左右转前后面经过穿过看到上下楼梯电梯门口沙发餐桌椅子床柜子")
        for i, char in enumerate(nav_chars):
            self.char_to_id[char] = len(self.char_to_id) + i

        print(f"词表大小：{len(self.char_to_id)}")

        # 模型配置（需要与 create_model 一致）
        d_model = 256  # 与模型配置一致

        # 生成样本
        for i in range(num_samples):
            # 随机生成指令（10-30 个字）
            instr_len = random.randint(10, 30)
            instr_ids = [self.char_to_id.get(random.choice(nav_chars), 1)
                         for _ in range(instr_len)]

            # 随机生成视觉特征
            num_views = 36
            visual_feat = [random.gauss(0, 1) for _ in range(num_views * self.feature_dim)]

            # 随机生成候选方向（使用 d_model 维度，不是 feature_dim）
            num_candidates = 36
            candidate_dirs = []
            for _ in range(num_candidates):
                cand = [random.gauss(0, 1) for _ in range(d_model)]  # 使用 d_model 维度
                candidate_dirs.append(cand)

            # 随机目标动作
            target = random.randint(0, num_candidates - 1)

            self.data.append(VLNExample(
                path_id=f"synth_{i:04d}",
                instruction=f"模拟指令_{i}",
                instruction_ids=instr_ids,
                visual_features=visual_feat,
                candidate_directions=candidate_dirs,
                target_action=target
            ))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> VLNExample:
        return self.data[idx]

    def get_vocab_size(self) -> int:
        return len(self.char_to_id)


def collate_fn(batch: List[VLNExample]) -> Dict[str, torch.Tensor]:
    """
    批处理 collate 函数
    """
    instructions = []
    instruction_masks = []
    visual_features = []
    candidate_directions = []
    target_actions = []

    for item in batch:
        # 指令（添加 <cls> 和 <sep>）
        instr_ids = torch.tensor([2] + item.instruction_ids + [3], dtype=torch.long)
        instructions.append(instr_ids)

        # mask (True 表示 padding 位置)
        mask = torch.zeros(len(instr_ids), dtype=torch.bool)
        mask[0] = False  # <cls>
        mask[-1] = False  # <sep>
        instruction_masks.append(mask)

        # 视觉特征
        visual = torch.tensor(item.visual_features, dtype=torch.float32)
        visual = visual.view(-1, 2048)  # [num_views, feature_dim]
        visual_features.append(visual)

        # 候选方向
        cand = torch.tensor(item.candidate_directions, dtype=torch.float32)
        candidate_directions.append(cand)

        # 目标动作
        target_actions.append(torch.tensor(item.target_action, dtype=torch.long))

    # padding
    instructions_padded = pad_sequence(instructions, batch_first=True, padding_value=0)
    instruction_masks_padded = pad_sequence(instruction_masks, batch_first=True, padding_value=True)
    visual_features_padded = pad_sequence(visual_features, batch_first=True, padding_value=0)
    candidate_directions_padded = pad_sequence(candidate_directions, batch_first=True, padding_value=0)
    target_actions_tensor = torch.stack(target_actions)

    return {
        'instructions': instructions_padded,
        'instruction_mask': instruction_masks_padded,
        'visual_features': visual_features_padded,
        'candidate_directions': candidate_directions_padded,
        'target_action': target_actions_tensor
    }


# ============================================================
# 训练器
# ============================================================

class VLNTrainer:
    """VLN 训练器"""

    def __init__(self, model: VLNBaseline, train_loader: DataLoader,
                 val_loader: DataLoader = None, lr: float = 1e-4,
                 weight_decay: float = 1e-4, device: str = 'mps'):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )

        self.criterion = nn.CrossEntropyLoss()

        self.best_val_loss = float('inf')
        self.train_history = []

    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = output['loss']

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
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

        avg_loss = total_loss / num_batches

        print(f"  Validation Epoch {epoch}, Loss: {avg_loss:.4f}")

        self.scheduler.step(avg_loss)

        # 保存最佳模型
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best')

        return avg_loss

    def save_checkpoint(self, name: str = 'checkpoint'):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        path = f"/Users/tyrion/Projects/Papers/checkpoints/vln_{name}.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"  已保存模型：{path}")

    def train(self, num_epochs: int = 10):
        """训练多个 epoch"""
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        print(f"设备：{self.device}")
        print(f"训练批次数：{len(self.train_loader)}")
        print(f"模型参数：{count_parameters(self.model):,}")
        print()

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 40)

            train_loss = self.train_epoch(epoch)
            print(f"  Training Loss: {train_loss:.4f}")

            val_loss = self.validate(epoch)

            self.train_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳验证损失：{self.best_val_loss:.4f}")
        print("=" * 60)

        # 保存最终模型
        self.save_checkpoint('final')


# ============================================================
# 主函数
# ============================================================

def main():
    # 配置
    batch_size = 8
    num_epochs = 5
    lr = 1e-3
    # MPS 不支持某些 Transformer 操作，使用 CPU
    device = 'cpu'

    print(f"使用设备：{device}")

    # 创建数据集（使用模拟数据）
    print("\n创建数据集...")
    train_dataset = VLNDataset(num_samples=500)
    val_dataset = VLNDataset(num_samples=100)

    vocab_size = train_dataset.get_vocab_size()
    print(f"词表大小：{vocab_size}")

    # 确保词表大小足够大
    vocab_size = max(vocab_size, 100)  # 至少 100，防止索引越界

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 创建模型
    print("\n创建模型...")
    model = create_model(vocab_size=vocab_size, d_model=256)
    print(f"模型参数：{count_parameters(model):,}")

    # 创建训练器
    trainer = VLNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        device=device
    )

    # 开始训练
    trainer.train(num_epochs=num_epochs)

    # 保存训练历史
    history_file = "/Users/tyrion/Projects/Papers/checkpoints/training_history.json"
    with open(history_file, 'w') as f:
        json.dump(trainer.train_history, f, indent=2)
    print(f"\n训练历史已保存到：{history_file}")


if __name__ == "__main__":
    main()
