# VLN 基线模型复现方案

> 日期：2026-03-19
> 目标：复现 VLN 基线模型，建立性能基准

---

## 1. 候选模型调研

### 1.1 VLN 核心模型对比

| 模型 | Venue | 架构 | 代码可用性 | 复现难度 |
|------|-------|------|-----------|---------|
| **Speaker-Follower** | ECCV 2018 | Seq2Seq + Attention | ✅ 官方开源 | ⭐⭐ |
| **FAST-Drop** | ECCV 2019 | 强化学习 | ✅ | ⭐⭐⭐ |
| **VLN-BERT** | CVPR 2021 | Transformer | ✅ | ⭐⭐⭐⭐ |
| **HAMT** | ICCV 2021 | Transformer + 历史记忆 | ✅ | ⭐⭐⭐⭐⭐ |
| **DUET** | ECCV 2022 | Transformer + 图网络 | ✅ | ⭐⭐⭐⭐ |
| **MapGPT** | 2024 | Transformer + 地图 | ✅ | ⭐⭐⭐⭐⭐ |

### 1.2 复现推荐

**推荐模型：VLN-BERT**

理由：
1. ✅ **代码开源**：https://github.com/cshizhe/VLN-BERT
2. ✅ **经典架构**：Transformer 编码器 + 路径解码器
3. ✅ **适中难度**：比 HAMT 简单，比 Seq2Seq 先进
4. ✅ **可迁移性**：可作为跨语言迁移的基线

**备选模型：Speaker-Follower (Anderson 2018)**

理由：
1. ✅ **最简单**：LSTM + Attention
2. ✅ **官方代码成熟**：https://github.com/peteranderson/VLNCE
3. ⚠️ **性能较低**：适合作为下界基准

---

## 2. VLN-BERT 架构解析

### 2.1 模型结构

```
┌─────────────────────────────────────────────────────────┐
│                    VLN-BERT 架构                         │
├─────────────────────────────────────────────────────────┤
│  输入层                                                   │
│  - 指令嵌入：[CLS] + 指令 tokens + [SEP]                  │
│  - 视觉特征：[View] tokens (多个视角)                      │
│  - 位置编码：方向 + 距离                                  │
├─────────────────────────────────────────────────────────┤
│  Transformer 编码器 (6-12 层)                               │
│  - 自注意力：指令 - 视觉交互                               │
│  - 输出：[CLS] 表示 (状态嵌入)                            │
├─────────────────────────────────────────────────────────┤
│  动作解码器                                               │
│  - 候选动作嵌入：K 个候选方向                             │
│  - 注意力权重：P(action|state)                           │
│  - 输出：选择的动作                                      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心模块

| 模块 | 功能 | PyTorch 实现 |
|------|------|-------------|
| **指令编码器** | BERT-style Transformer | `nn.TransformerEncoder` |
| **视觉编码器** | CNN/ResNet 特征提取 | `torchvision.models.resnet` |
| **融合模块** | 跨模态注意力 | `nn.MultiheadAttention` |
| **动作预测** | 路径选择 | `nn.Linear + Softmax` |

---

## 3. 复现方案

### 3.1 简化版 VLN-BERT（推荐入门）

**目标**：用 1-2 周实现核心功能

**简化策略**：
1. 使用预训练 ResNet 提取视觉特征（不训练视觉部分）
2. 简化 Transformer 层数（6 层 → 2 层）
3. 使用 R2R 原始数据（不扩展）
4. 先在验证集上测试，不跑完整测试集

**预期性能**：
- 原始 VLN-BERT: SR~55%, SPL~50%
- 简化版目标：SR~45%, SPL~40%

### 3.2 完整复现（进阶）

**目标**：接近原论文性能

**需要资源**：
- GPU: 至少 1×RTX 3090/4090 或 A100
- 时间：2-4 周
- 存储：~100GB (MatterPort3D + 特征)

---

## 4. 实施步骤

### 阶段 1：环境准备（1-2 天）

```bash
# 创建环境
conda create -n vln python=3.9
conda activate vln

# 安装核心依赖
pip install torch torchvision
pip install transformers
pip install habitat-sim  # 如需仿真环境

# 安装 VLN-BERT 依赖
git clone https://github.com/cshizhe/VLN-BERT.git
cd VLN-BERT
pip install -r requirements.txt
```

### 阶段 2：数据准备（2-3 天）

```bash
# 下载 R2R 数据
# 1. 下载路径数据
wget https://raw.githubusercontent.com/peteranderson/VLNCE/master/data/R2R/train.json
wget https://raw.githubusercontent.com/peteranderson/VLNCE/master/data/R2R/val_seen.json
wget https://raw.githubusercontent.com/peteranderson/VLNCE/master/data/R2R/val_unseen.json

# 2. 下载视觉特征（预提取，约 50GB）
# 从 Google Drive 或官方源下载

# 3. 准备中文指令数据（我们已生成）
cp /Users/tyrion/Projects/Papers/data/generated_instructions.json ./data/chinese_train.json
```

### 阶段 3：模型实现（3-5 天）

**核心文件**：
```
vln_baseline/
├── model/
│   ├── vln_bert.py      # VLN-BERT 模型
│   ├── attention.py     # 注意力机制
│   └── encoder.py       # 指令/视觉编码器
├── data/
│   ├── dataset.py       # 数据加载
│   └── preproc.py       # 预处理
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
└── config.py            # 配置
```

### 阶段 4：训练与评估（2-3 天）

```bash
# 训练
python train.py \
  --model vln_bert \
  --data data/r2r_train.json \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4

# 评估
python evaluate.py \
  --model checkpoints/vln_bert_best.pt \
  --data data/r2r_val.json
```

---

## 5. 评估指标

| 指标 | 说明 | 计算公式 |
|------|------|----------|
| **SR (成功率)** | 到达目标的比率 | #success / #total |
| **SPL (成功率×路径长度)** | 效率加权成功率 | Σ success_i × min(d_i, l_i) / d_i |
| **SDTW (动态时间弯曲)** | 路径相似度 | 归一化 DTW 距离 |
| **Oracle** | 上限估计 | 如果任何候选正确则计为成功 |

---

## 6. 预期时间线

| 阶段 | 时间 | 里程碑 |
|------|------|--------|
| 环境 + 数据 | 第 1 周 | 数据加载完成 |
| 模型实现 | 第 2 周 | 代码跑通 |
| 训练调试 | 第 3 周 | SR>30% |
| 优化 + 分析 | 第 4 周 | SR>40%, 完成报告 |

---

## 7. 风险与应对

| 风险 | 可能性 | 影响 | 应对 |
|------|--------|------|------|
| 数据下载失败 | 中 | 中 | 使用镜像源/手动下载 |
| GPU 内存不足 | 高 | 中 | 减小 batch size/梯度累积 |
| 训练不收敛 | 中 | 高 | 检查学习率/使用预训练权重 |
| 性能远低于预期 | 中 | 中 | 对比官方代码/debug |

---

## 8. 参考资源

### 8.1 官方代码库

- **VLN-BERT**: https://github.com/cshizhe/VLN-BERT
- **HAMT**: https://github.com/cshizhe/HAMT
- **VLNCE**: https://github.com/facebookresearch/vln-ce
- **Speaker-Follower**: https://github.com/peteranderson/VLNCE

### 8.2 教程与文档

- VLN 入门教程：https://github.com/peteranderson/VLNCE/blob/master/docs/README.md
- Habitat 文档：https://aihabitat.org/docs/

---

## 9. 下一步行动

### 立即可执行

- [ ] 确认 GPU 资源可用性
- [ ] 克隆 VLN-BERT 代码库
- [ ] 测试数据加载

### 本周完成

- [ ] 搭建训练环境
- [ ] 下载并预处理数据
- [ ] 运行官方代码验证环境

---

*方案制定日期：2026-03-19*
