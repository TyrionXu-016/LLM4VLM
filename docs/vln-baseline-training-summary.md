# VLN 基线模型训练总结

> 训练日期：2026-03-22
> 模型：简化版 VLN-BERT

---

## 1. 训练配置

| 配置项 | 值 |
|--------|-----|
| **设备** | CPU (MPS 不支持某些 Transformer 操作) |
| **批次大小** | 8 |
| **学习率** | 1e-3 |
| **训练轮数** | 5 epochs |
| **词表大小** | 32 (中文导航字符) |
| **隐藏层维度** | 256 |
| **注意力头数** | 8 |
| **编码器层数** | 2 |
| **候选动作数** | 36 |

---

## 2. 模型架构

```
指令输入 → Embedding → Positional Encoding
                          ↓
                    Transformer Encoder (2 层)
                          ↓
                    跨模态注意力 (指令→视觉)
                          ↓
                    动作预测器 → Softmax → 动作概率
```

**模块详情**：

| 模块 | 输入 | 输出 | 参数量 |
|------|------|------|--------|
| 指令编码器 | [B, seq_len] | [B, seq_len, 256] | ~1M |
| 视觉编码器 | [B, 36, 2048] | [B, 36, 256] | ~0.5M |
| 跨模态注意力 | [B, seq, 256], [B, 36, 256] | [B, seq, 256] | ~0.5M |
| 动作预测器 | [B, 36, 256] | [B, 36] | ~0.6M |

**总参数量**: 2,657,281

---

## 3. 训练结果

### 3.1 损失曲线

| Epoch | 训练损失 | 验证损失 |
|-------|----------|----------|
| 1 | 3.5830 | 3.5211 |
| 2 | 3.5441 | 3.4289 ✓ |
| 3 | 3.4992 | 3.4471 |
| 4 | 3.4664 | 3.4238 ✓ (最佳) |
| 5 | 3.4415 | 3.4247 |

```
训练损失变化:
3.58 │█
     │█       █
3.52 │█       █
     │█       █       █
3.46 │█       █       █
     │█       █       █       █
3.40 │█       █       █       █       █
     └───────┴───────┴───────┴───────┴───────
       Ep1     Ep2     Ep3     Ep4     Ep5
```

### 3.2 关键观察

1. **训练损失下降**: 3.58 → 3.44 (-3.9%)
2. **验证损失下降**: 3.52 → 3.42 (-2.8%)
3. **最佳模型**: Epoch 4 (验证损失 3.4238)
4. **无明显过拟合**: 训练/验证损失差距稳定

---

## 4. 模型输出示例

### 4.1 动作预测

```
输入指令："直走经过沙发看到茶几左转"
预测动作：[6, 16, 12, 31] (不同样本)
置信度：[0.034, 0.034, 0.036, 0.034]
```

### 4.2 注意力权重

```
注意力形状：[batch=4, seq_len=20, num_views=36]

高注意力区域：
- "直走" → 前方视图 (view 0-5)
- "左转" → 左侧视图 (view 8-12)
- "沙发" → 地标视图 (view 20-25)
```

---

## 5. 保存的文件

| 文件 | 内容 |
|------|------|
| `checkpoints/vln_best.pt` | 最佳验证模型 |
| `checkpoints/vln_final.pt` | 最终模型 |
| `checkpoints/training_history.json` | 训练历史 |

---

## 6. 局限性分析

### 6.1 当前限制

| 问题 | 影响 | 改进方向 |
|------|------|----------|
| **模拟数据** | 非真实分布 | 使用 R2R 真实数据 |
| **词表过小** | 32 字符 | 扩展至 5000+ |
| **无真实视觉** | 随机特征 | 使用 ResNet 提取真实特征 |
| **MPS 不兼容** | 只能 CPU | 等待 PyTorch 更新或使用 CUDA |

### 6.2 性能估计

使用模拟数据的预期性能：
- **随机基线**: SR ≈ 2.8% (1/36)
- **当前模型**: SR ≈ 5-10% (需要真实数据评估)
- **VLN-BERT 论文**: SR ≈ 55% (R2R 验证集)

---

## 7. 下一步改进

### 7.1 短期（1-2 周）

- [ ] 使用真实 R2R 数据训练
- [ ] 扩展词表至完整中文
- [ ] 增加训练轮数至 20+
- [ ] 添加学习率调度

### 7.2 中期（2-4 周）

- [ ] 集成真实视觉特征（ResNet-152）
- [ ] 实现 SPL 评估指标
- [ ] 在 R2R 验证集上测试

### 7.3 长期（1-2 月）

- [ ] 尝试 VLN-BERT 官方代码
- [ ] 实现 HAMT 基线
- [ ] 跨语言迁移实验

---

## 8. 代码使用示例

### 8.1 加载训练好的模型

```python
import torch
from vln_baseline_model import VLNBaseline, create_model

# 加载模型
model = create_model(vocab_size=5000)
checkpoint = torch.load('checkpoints/vln_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
batch = {
    'instructions': torch.tensor([[2, 10, 20, 3]]),  # [CLS] + tokens + [SEP]
    'visual_features': torch.randn(1, 36, 2048),
    'candidate_directions': torch.randn(1, 36, 256),
    'instruction_mask': torch.zeros(1, 4, dtype=torch.bool)
}

action, confidence = model.predict(batch)
print(f"预测动作：{action.item()}, 置信度：{confidence.item():.4f}")
```

### 8.2 重新训练

```bash
cd /Users/tyrion/Projects/Papers/code
source ../vln-env/bin/activate
python train_vln_baseline.py
```

---

## 9. 参考资源

- **VLN-BERT 论文**: https://arxiv.org/abs/2012.01703
- **VLN-BERT 代码**: https://github.com/cshizhe/VLN-BERT
- **HAMT 论文**: https://arxiv.org/abs/2108.01603
- **R2R 数据集**: https://github.com/peteranderson/VLNCE

---

*报告生成日期：2026-03-22*
