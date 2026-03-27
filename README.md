# LLM4VLM: Large Language Models for Zero-Shot Cross-Lingual Vision-and-Language Navigation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

> 使用大语言模型生成中文导航指令，实现跨语言视觉 - 语言导航

## 📖 目录

- [简介](#简介)
- [主要贡献](#主要贡献)
- [方法概述](#方法概述)
- [安装](#安装)
- [快速开始](#快速开始)
- [数据准备](#数据准备)
- [训练模型](#训练模型)
- [评估模型](#评估模型)
- [可视化](#可视化)
- [模型结果](#模型结果)
- [项目结构](#项目结构)
- [引用](#引用)
- [许可证](#许可证)

---

## 简介

视觉 - 语言导航（VLN）要求智能体遵循自然语言指令在视觉环境中导航。尽管英语 VLN 研究取得了显著进展，但跨语言迁移（特别是中文）仍未得到充分探索。

本项目系统地研究了如何利用大语言模型（LLM）进行中文 VLN 指令生成和跨语言迁移，实现了：
- **LLM 指令生成**：76.7% 优秀率（vs 机器翻译 20%）
- **VLN 基线模型**：62.0% 成功率（超越 VLN-BERT 基线 12.7%）
- **ResNet 特征增强**：训练损失降低 22%，训练时间减少 83%

## 主要贡献

1. **首个中文 VLN 指令生成方法**：基于 LLM 的跨语言 VLN 数据创建新范式
2. **对照实验验证**：LLM 生成 vs 机器翻译，76.7% vs 20% 优秀率
3. **强基线模型**：62.0% 成功率，配备 ResNet-50 视觉特征
4. **全面消融研究**：视觉特征、训练策略、指令质量分析
5. **开源代码和数据**：促进跨语言 VLN 研究

## 方法概述

```
┌─────────────────┐
│  Path Input     │
│  [起点]→客厅    │
│  →走廊→厨房     │
└────────┬────────┘
         │ 路径描述
         ▼
┌─────────────────┐
│  LLM 生成器      │
│  (Qwen-3.5)     │
└────────┬────────┘
         │ 生成
         ▼
┌─────────────────┐
│ 中文指令         │
│ "直走 3 米，然后左转"│
└────────┬────────┘
         │ Tokenize
         ▼
┌─────────────────────────────┐
│    VLN-BERT 基线模型         │
│  ┌───────┐    ┌──────────┐  │
│  │指令编码 │    │ 视觉编码 │  │
│  │(256-d)│    │(ResNet-50)│ │
│  └───┬───┘    └────┬─────┘  │
│      └─────┬───────┘        │
│    跨模态注意力融合          │
└─────────────┬───────────────┘
              │ 预测
              ▼
┌─────────────────┐
│  动作输出        │
│  Softmax(36)    │
│  最佳：左转      │
└─────────────────┘
```

## 安装

### 环境要求

- Python 3.9+
- PyTorch 2.x
- macOS (Apple Silicon) 或 Linux

### 创建虚拟环境

```bash
# 克隆仓库
git clone https://github.com/your-username/llm4vlm.git
cd llm4vlm

# 创建虚拟环境
python -m venv vln-env
source vln-env/bin/activate  # Linux/macOS
# 或 vln-env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 生成中文导航指令

```bash
python code/generate_chinese_instructions.py \
    --input data/sample_paths.json \
    --output data/generated_instructions.json \
    --api-key YOUR_API_KEY
```

### 2. 准备训练数据

```bash
python code/generate_r2r_enhanced_data.py \
    --instructions data/generated_instructions.json \
    --output data/r2r_enhanced/train.json
```

### 3. 训练模型

```bash
python code/train_vln_r2r_enhanced.py \
    --data data/r2r_enhanced/train.json \
    --val-data data/r2r_enhanced/val.json \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-4 \
    --output checkpoints/vln_r2r_best.pt
```

### 4. 评估模型

```bash
python code/evaluate_vln_model.py \
    --model checkpoints/vln_r2r_best.pt \
    --data data/r2r_enhanced/val.json \
    --output data/evaluation_results.json
```

## 数据准备

### 数据集结构

```
data/
├── r2r_enhanced/
│   ├── train.json          # 训练数据（LLM 生成）
│   ├── val.json            # 验证数据
│   └── vocabulary.json     # 中文词表（97 字符）
├── r2r_raw/                # 原始 R2R 数据
├── sample_paths.json       # 路径样本
└── generated_instructions.json  # LLM 生成指令
```

### 数据格式

每个训练样本包含：
- `path_id`: 路径唯一标识
- `instruction`: 中文导航指令
- `instruction_ids`: tokenized 指令 IDs
- `visual_features`: ResNet-50 视觉特征 (2048-d)
- `candidate_directions`: 候选方向特征
- `target_action`: 目标动作 (0-35)

## 训练模型

### 模型架构

| 组件 | 配置 |
|------|------|
| 指令编码器 | Transformer Encoder (2 层) |
| 视觉编码器 | ResNet-50 + FC (2048→256) |
| 注意力头数 | 8 |
| 隐藏维度 | 256 |
| 词表大小 | 97 字符 |

### 训练超参数

| 超参数 | 值 |
|--------|-----|
| Batch Size | 16 (有效 32，梯度累积 2 步) |
| 学习率 | 1e-4 (warmup + 余弦退火) |
| Warmup 轮数 | 3 |
| 最大轮数 | 20 |
| 早停耐心值 | 5 |
| Dropout | 0.1 |
| 优化器 | AdamW |

### 训练进度监控

训练过程中会自动保存：
- 每个 epoch 的训练/验证损失
- 最佳模型检查点
- 训练历史 JSON

## 评估模型

### 评估指标

- **SR (成功率)**: 距离目标 < 3m 的比例
- **SPL (路径长度加权成功率)**: 效率加权成功率
- **Oracle SR**: 轨迹中任意点到达目标的比例
- **DTW**: 动态时间规整距离

### 运行评估

```bash
python code/evaluate_vln_model.py \
    --model checkpoints/vln_r2r_best.pt \
    --val-data data/r2r_enhanced/val.json \
    --output data/evaluation/
```

### 预期结果

| 模型 | SR | SPL | Oracle SR |
|------|-----|-----|-----------|
| VLN-BERT (论文) | ~55% | ~50% | - |
| **我们的模型** | **62.0%** | **61.8%** | **69.0%** |

## 可视化

### 训练曲线

```bash
python code/generate_paper_figures.py
```

生成：
- 训练损失对比图
- SR/SPL 指标对比图
- 消融实验结果图
- 置信度分布图

### 注意力可视化

```bash
python code/visualize_attention.py
```

生成跨模态注意力权重可视化，展示语言 - 视觉对齐模式。

## 模型结果

### 主要结果

| 指标 | 本模型 | VLN-BERT 论文 | 改进 |
|------|--------|---------------|------|
| **SR** | **62.0%** | ~55% | +12.7% |
| **SPL** | **61.8%** | ~50% | +23.6% |
| **Oracle SR** | **69.0%** | - | - |

### 消融研究

| 特征类型 | 验证损失 | 训练时间 |
|----------|----------|----------|
| 随机高斯 | 3.37 | 6 分钟 |
| **ResNet-50** | **2.63 (-22%)** | **1 分钟 (-83%)** |

### 按指令类型分析

| 类型 | 样本数 | 成功率 |
|------|--------|--------|
| 左转 | 104 | 58.7% |
| 右转 | 96 | 65.6% |
| 直走 | 109 | 68.8% |
| 楼梯 | 22 | 77.3% |

## 项目结构

```
llm4vlm/
├── README.md                 # 本文件
├── LICENSE                   # MIT 许可证
├── requirements.txt          # Python 依赖
├── code/                     # 源代码
│   ├── vln_baseline_model.py    # VLN 模型定义
│   ├── train_vln_r2r_enhanced.py # 训练脚本
│   ├── evaluate_vln_model.py    # 评估脚本
│   ├── generate_r2r_enhanced_data.py # 数据生成
│   ├── visualize_attention.py   # 注意力可视化
│   └── ...
├── data/                     # 数据目录
│   ├── r2r_enhanced/         # 增强 R2R 数据
│   ├── r2r_raw/              # 原始 R2R 数据
│   └── evaluation/           # 评估结果
├── checkpoints/              # 模型检查点
│   ├── vln_r2r_best.pt       # 最佳模型
│   └── training_history.json # 训练历史
└── paper/                    # 论文和图表
    ├── vln-llm-paper-draft.md    # 论文草稿
    ├── figures/                  # 图表文件
    └── supplementary_materials.md # 补充材料
```

## 引用

如果您使用本项目的代码或数据，请引用我们的论文：

```bibtex
@article{llm4vlm2024,
  title={LLM4VLM: Large Language Models for Zero-Shot Cross-Lingual Vision-and-Language Navigation},
  author={Author Name and Advisor Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

- R2R 数据集：Anderson et al., 2018
- VLN-BERT：Hong et al., 2021
- Qwen LLM：阿里云通义千问

## 联系方式

如有问题或建议，请提交 Issue 或联系作者。
