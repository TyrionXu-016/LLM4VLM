"""
简化版 VLN 基线模型

基于 VLN-BERT 架构的简化实现，用于教学和理解核心原理
包含：指令编码、视觉注意力、路径选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """位置编码（用于 Transformer）"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, batch, d_model]"""
        return x + self.pe[:x.size(0)]


class InstructionEncoder(nn.Module):
    """指令编码器（BERT-style Transformer）"""

    def __init__(self, vocab_size: int, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 2,
                 dropout: float = 0.1, max_len: int = 100):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.d_model = d_model

    def forward(self, instructions: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            instructions: [batch, seq_len] - 指令 token IDs
            mask: [batch, seq_len] - padding mask

        Returns:
            encoded: [batch, seq_len, d_model] - 编码后的指令表示
        """
        embedded = self.embedding(instructions) * math.sqrt(self.d_model)
        encoded = self.pos_encoder(embedded)

        if mask is not None:
            # Transformer 需要 src_key_padding_mask
            output = self.transformer(encoded, src_key_padding_mask=mask)
        else:
            output = self.transformer(encoded)

        return output


class VisualEncoder(nn.Module):
    """视觉编码器（简化版，使用预提取特征）"""

    def __init__(self, feature_dim: int = 2048, d_model: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, num_views, feature_dim] - 预提取的视觉特征

        Returns:
            encoded: [batch, num_views, d_model] - 编码后的视觉表示
        """
        return self.projection(features)


class CrossModalAttention(nn.Module):
    """跨模态注意力（指令 - 视觉融合）"""

    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, instruction_emb: torch.Tensor,
                visual_emb: torch.Tensor,
                instruction_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            instruction_emb: [batch, seq_len, d_model] - 指令嵌入
            visual_emb: [batch, num_views, d_model] - 视觉嵌入
            instruction_mask: [batch, seq_len] - 指令 padding mask

        Returns:
            attended_instruction: [batch, seq_len, d_model]
            attention_weights: [batch, seq_len, num_views]
        """
        # 指令作为 query，视觉作为 key/value
        attended, attn_weights = self.attention(
            query=instruction_emb,
            key=visual_emb,
            value=visual_emb,
            key_padding_mask=None
        )

        attended = self.norm(instruction_emb + self.dropout(attended))

        return attended, attn_weights


class ActionPredictor(nn.Module):
    """动作预测器（路径选择）"""

    def __init__(self, d_model: int = 256, num_candidates: int = 36):
        super().__init__()

        # [CLS] token 用于聚合状态信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 融合指令和视觉信息
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 预测动作
        self.action_head = nn.Linear(d_model, 1)

        self.num_candidates = num_candidates

    def forward(self, instruction_emb: torch.Tensor,
                visual_emb: torch.Tensor,
                candidate_directions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            instruction_emb: [batch, seq_len, d_model] - 注意力后的指令
            visual_emb: [batch, num_views, d_model] - 视觉特征
            candidate_directions: [batch, num_candidates, d_model] - 候选方向特征

        Returns:
            action_probs: [batch, num_candidates] - 动作概率分布
        """
        # 使用 [CLS] 聚合指令信息
        batch_size = instruction_emb.size(0)
        cls_expanded = self.cls_token.expand(batch_size, -1, -1)

        # 聚合指令表示（简单平均）
        instruction_mean = instruction_emb.mean(dim=1)  # [batch, d_model]

        # 融合指令和候选方向
        candidate_expanded = candidate_directions  # [batch, num_candidates, d_model]

        # 为每个候选方向计算分数
        scores = []
        for i in range(candidate_directions.size(1)):
            cand_feat = candidate_directions[:, i, :]  # [batch, d_model]
            fused = torch.cat([instruction_mean, cand_feat], dim=-1)  # [batch, 2*d_model]
            score = self.action_head(self.fusion(fused))  # [batch, 1] (logit)
            scores.append(score)

        scores = torch.cat(scores, dim=-1)  # [batch, num_candidates]
        # 返回 logits；softmax 在外部根据需要计算
        return scores


class VLNBaseline(nn.Module):
    """
    简化版 VLN 基线模型

    架构:
        指令 → 指令编码器 → Transformer → [CLS]
                                    ↓
        视觉 → 视觉编码器 → 跨模态注意力 → 动作预测
    """

    def __init__(self, vocab_size: int, feature_dim: int = 2048,
                 d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 2, dropout: float = 0.1,
                 num_candidates: int = 36):
        super().__init__()

        self.instruction_encoder = InstructionEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dropout=dropout
        )

        self.visual_encoder = VisualEncoder(
            feature_dim=feature_dim,
            d_model=d_model
        )

        self.cross_attention = CrossModalAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )

        self.action_predictor = ActionPredictor(
            d_model=d_model,
            num_candidates=num_candidates
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: 包含以下键的字典
                - instructions: [batch, seq_len] - 指令 token IDs
                - visual_features: [batch, num_views, feature_dim] - 视觉特征
                - candidate_directions: [batch, num_candidates, d_model] - 候选方向
                - instruction_mask: [batch, seq_len] - 指令 padding mask
                - target_action: [batch] - 目标动作（用于训练）

        Returns:
            output: 包含以下键的字典
                - action_probs: [batch, num_candidates] - 动作概率
                - loss: 标量 - 交叉熵损失（如果提供 target_action）
        """
        # 编码指令
        instruction_emb = self.instruction_encoder(
            batch['instructions'],
            mask=batch.get('instruction_mask')
        )

        # 编码视觉
        visual_emb = self.visual_encoder(batch['visual_features'])

        # 跨模态注意力
        attended_instruction, attn_weights = self.cross_attention(
            instruction_emb,
            visual_emb,
            instruction_mask=batch.get('instruction_mask')
        )

        # 预测动作（logits -> probs）
        action_logits = self.action_predictor(
            attended_instruction,
            visual_emb,
            batch['candidate_directions']
        )
        action_probs = F.softmax(action_logits, dim=-1)

        output = {
            'action_probs': action_probs,
            'attention_weights': attn_weights
        }

        # 计算损失（训练时）
        if 'target_action' in batch:
            # 交叉熵期望 logits（未 softmax），避免 softmax 两次导致训练失效
            loss = F.cross_entropy(action_logits, batch['target_action'])
            output['loss'] = loss

        return output

    def predict(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测动作（推理模式）

        Returns:
            action: [batch] - 预测的动作索引
            confidence: [batch] - 置信度
        """
        self.eval()
        with torch.no_grad():
            output = self(batch)
            probs = output['action_probs']
            confidence, action = torch.max(probs, dim=-1)
            return action, confidence


# ============================================================
# 训练辅助函数
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(vocab_size: int, **kwargs) -> VLNBaseline:
    """创建模型（带默认配置）"""
    default_config = {
        'vocab_size': vocab_size,
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 2,
        'dropout': 0.1,
        'num_candidates': 36
    }
    default_config.update(kwargs)
    return VLNBaseline(**default_config)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    # 模型测试
    print("=" * 60)
    print("VLN 基线模型测试")
    print("=" * 60)

    # 创建模型
    vocab_size = 5000
    batch_size = 4
    seq_len = 20
    num_views = 36
    feature_dim = 2048
    num_candidates = 36

    model = create_model(vocab_size=vocab_size)
    print(f"\n模型参数数量：{count_parameters(model):,}")

    # 创建测试数据
    batch = {
        'instructions': torch.randint(1, vocab_size, (batch_size, seq_len)),
        'visual_features': torch.randn(batch_size, num_views, feature_dim),
        'candidate_directions': torch.randn(batch_size, num_candidates, 256),
        'instruction_mask': torch.zeros(batch_size, seq_len, dtype=torch.bool),
        'target_action': torch.randint(0, num_candidates, (batch_size,))
    }

    # 前向传播
    print("\n前向传播测试...")
    output = model(batch)

    print(f"动作概率形状：{output['action_probs'].shape}")
    print(f"注意力权重形状：{output['attention_weights'].shape}")
    print(f"损失：{output['loss'].item():.4f}")

    # 预测
    print("\n预测测试...")
    action, confidence = model.predict(batch)
    print(f"预测动作：{action.tolist()}")
    print(f"置信度：{confidence.tolist()}")

    print("\n✓ 模型测试通过!")
