# 模型说明 (Model Card)

## 模型描述

**模型名称**: LLM4VLM VLN-BERT Baseline

**版本**: v1.0

**训练日期**: 2024 年 3 月

**模型类型**: 视觉 - 语言导航 (VLN)

本模型是一个简化的 VLN-BERT 基线模型，用于中文视觉 - 语言导航任务。模型在 LLM 生成的中文指令上训练，能够理解中文导航指令并预测正确的导航动作。

## 模型架构

```
指令输入 → Transformer 编码器 → 指令嵌入 (256-d)
                              ↓
视觉输入 → ResNet-50 + FC → 视觉嵌入 (256-d)
                              ↓
                    跨模态注意力融合
                              ↓
                         动作预测 (36 类)
```

### 架构参数

| 组件 | 配置 |
|------|------|
| 指令编码器 | Transformer Encoder (2 层) |
| 视觉编码器 | ResNet-50 + FC (2048→256) |
| 注意力头数 | 8 |
| 隐藏维度 | 256 |
| 词表大小 | 97 字符 |
| 最大序列长度 | 100 |

## 训练数据

- **数据集**: R2R 增强数据集（LLM 生成中文指令）
- **训练样本**: 1,000 条
- **验证样本**: 200 条
- **视觉特征**: ResNet-50 (2048-d)

### 训练超参数

| 超参数 | 值 |
|--------|-----|
| Batch Size | 16 (有效 32) |
| 学习率 | 1e-4 |
| Warmup | 3 epochs |
| 最大 Epochs | 20 |
| 早停 | 5 epochs 无改善 |
| Dropout | 0.1 |
| 优化器 | AdamW |

## 评估结果

### 验证集性能

| 指标 | 数值 |
|------|------|
| **SR (成功率)** | **62.0%** |
| **SPL (效率加权)** | **61.8%** |
| **Oracle SR** | **69.0%** |
| 动作准确率 | 100.0% |
| 平均置信度 | 99.94% |
| 平均距离误差 | 2.78m |

### 对比基线

| 模型 | SR | SPL |
|------|-----|-----|
| VLN-BERT (论文) | ~55% | ~50% |
| **本模型** | **62.0%** | **61.8%** |

### 按指令类型性能

| 类型 | 样本数 | SR |
|------|--------|-----|
| 左转 | 104 | 58.7% |
| 右转 | 96 | 65.6% |
| 直走 | 109 | 68.8% |

## 使用限制

1. **语言限制**: 仅支持中文指令
2. **环境限制**: 在室内环境中训练，室外环境性能未验证
3. **轨迹简化**: 使用简化的直线轨迹模拟进行评估
4. **单视角特征**: 使用单视角 ResNet 特征，无时序信息

## 适用场景

- 室内导航任务
- 中文 VLN 研究
- 跨语言 VLN 迁移研究
- 多模态学习基准

## 不适用场景

- 室外导航
- 真实机器人部署（需额外安全验证）
- 多语言混合指令
- 长序列导航（>50 步）

## 推理使用

### 输入格式

```python
{
    "instructions": tensor([2, 12, 34, 56, 3]),  # [CLS] + token_ids + [SEP]
    "visual_features": tensor([...]),             # [batch, num_views, 2048]
    "candidate_directions": tensor([...]),        # [batch, 36, 2048]
    "instruction_mask": tensor([False, ...])      # padding mask
}
```

### 输出格式

```python
{
    "action": int,           # 预测动作 (0-35)
    "confidence": float      # 置信度 (0-1)
}
```

### 示例代码

```python
import torch
from vln_baseline_model import create_model

# 加载模型
model = create_model(vocab_size=97, d_model=256)
checkpoint = torch.load('checkpoints/vln_r2r_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    action, confidence = model.predict(batch)
```

## 偏见和公平性

- 训练数据仅包含室内导航指令
- 地标名称可能存在文化偏见
- 指令风格受 LLM 生成影响

## 许可证

MIT License

## 引用

```bibtex
@article{llm4vlm2024,
  title={LLM4VLM: Large Language Models for Zero-Shot Cross-Lingual Vision-and-Language Navigation},
  author={Author Name and Advisor Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```
