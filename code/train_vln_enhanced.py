"""
增强的 VLN 基线训练脚本

改进:
1. 增加训练轮数 (20+ epochs)
2. 添加早停机制
3. 学习率预热和衰减
4. 梯度累积
5. 更详细的训练日志
"""

import os
import sys
import json
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../LLM4VLM
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from vln_baseline_model import VLNBaseline, create_model, count_parameters


# ============================================================
# 数据集 (增强版)
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


class VLNDataset(Dataset):
    """VLN 数据集"""

    def __init__(self, data_file: str = None, vocab_size: int = 5000,
                 num_samples: int = 1000, feature_dim: int = 2048,
                 use_real_data: bool = False):
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.use_real_data = use_real_data
        self.data = []

        # 扩展词表 (中文导航常用字)
        self._build_vocab()

        if data_file and os.path.exists(data_file) and use_real_data:
            self._load_real_data(data_file)
        else:
            self._generate_synthetic_data(num_samples)

    def _build_vocab(self):
        """构建中文导航词表"""
        self.char_to_id = {
            '<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3
        }

        # 扩展中文导航词汇
        nav_chars = (
            "直走左右转前后面经过穿过看到上下楼梯电梯门口沙发餐桌椅子床柜子"
            "客厅卧室厨房卫生间阳台花园走廊过道大厅房间入口出口尽头旁边对面"
            "第一第二个三个几个几步米远近处远处近处这里那里这边那边"
            "进去出来上去下来过去过来回到停留继续前进后退"
            "红色蓝色绿色黄色白色黑色大小新旧老干湿"
        )
        for char in nav_chars:
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)

        print(f"词表大小：{len(self.char_to_id)}")

    def _generate_synthetic_data(self, num_samples: int):
        """生成模拟训练数据"""
        random.seed(42)
        d_model = 256

        nav_words = ["直走", "左转", "右转", "经过", "穿过", "看到", "上楼", "下楼",
                     "沙发", "餐桌", "椅子", "床", "柜子", "厨房", "卧室", "门口"]

        for i in range(num_samples):
            # 生成更自然的指令
            num_words = random.randint(5, 15)
            instruction_words = [random.choice(nav_words) for _ in range(num_words)]
            instruction = ''.join(instruction_words)

            # 转换为 ID
            instr_ids = []
            for word in instruction_words:
                for char in word:
                    instr_ids.append(self.char_to_id.get(char, 1))

            # 视觉特征
            num_views = 36
            visual_feat = [random.gauss(0, 0.5) for _ in range(num_views * self.feature_dim)]

            # 候选方向
            num_candidates = 36
            candidate_dirs = []
            for _ in range(num_candidates):
                cand = [random.gauss(0, 0.5) for _ in range(d_model)]
                candidate_dirs.append(cand)

            # 目标动作 (让某些模式可学习)
            if '左转' in instruction:
                target = random.randint(8, 12)
            elif '右转' in instruction:
                target = random.randint(24, 28)
            elif '上楼' in instruction:
                target = random.randint(0, 4)
            else:
                target = random.randint(0, num_candidates - 1)

            self.data.append(VLNExample(
                path_id=f"synth_{i:04d}",
                instruction=instruction,
                instruction_ids=instr_ids[:50],  # 限制最大长度
                visual_features=visual_feat,
                candidate_directions=candidate_dirs,
                target_action=target
            ))

    def _load_real_data(self, data_file: str):
        """加载真实数据"""
        # TODO: 实现 R2R 数据加载
        print(f"加载真实数据：{data_file}")

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

    for item in batch:
        # 指令
        instr_ids = torch.tensor([2] + item.instruction_ids + [3], dtype=torch.long)
        instructions.append(instr_ids)

        # mask
        mask = torch.zeros(len(instr_ids), dtype=torch.bool)
        instruction_masks.append(mask)

        # 视觉特征
        visual = torch.tensor(item.visual_features, dtype=torch.float32)
        visual = visual.view(-1, 2048)
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
# 增强训练器
# ============================================================

class EnhancedVLNTrainer:
    """增强的 VLN 训练器"""

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
            if batch_idx % 20 == 0:
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
        path = CHECKPOINT_DIR / f"vln_{name}.pt"
        os.makedirs(path.parent, exist_ok=True)
        torch.save(checkpoint, str(path))
        print(f"  已保存模型：{path}")

    def train(self, num_epochs: int = 20):
        """训练"""
        print("=" * 70)
        print("增强版 VLN 训练")
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
        history_file = CHECKPOINT_DIR / "training_history.json"
        os.makedirs(history_file.parent, exist_ok=True)
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2)


# ============================================================
# 主函数
# ============================================================

def main():
    # 增强配置
    batch_size = 16
    num_epochs = 20
    lr = 1e-4
    grad_accum_steps = 2  # 有效 batch_size = 16 * 2 = 32
    patience = 5
    warmup_epochs = 3
    device = 'cpu'

    print(f"使用设备：{device}")
    print(f"有效批次大小：{batch_size * grad_accum_steps}")

    # 创建数据集
    print("\n创建数据集...")
    train_dataset = VLNDataset(num_samples=2000)
    val_dataset = VLNDataset(num_samples=500)

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
    trainer = EnhancedVLNTrainer(
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
