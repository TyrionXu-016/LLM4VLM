# LLM4VLM 代码目录说明

## 核心代码

### 模型定义
- `vln_baseline_model.py` - VLN-BERT 基线模型定义
  - InstructionEncoder: 指令编码器
  - VisualEncoder: 视觉编码器
  - CrossModalAttention: 跨模态注意力
  - ActionPredictor: 动作预测器
  - VLNBaseline: 完整模型

### 训练脚本
- `train_vln_r2r_enhanced.py` - 主训练脚本（推荐）
  - 支持 R2R 增强数据
  - 梯度累积
  - 学习率调度（warmup + cosine）
  - 早停机制
  - 模型检查点保存

- `train_vln_baseline.py` - 基础训练脚本
- `train_vln_enhanced.py` - 增强版训练脚本

### 数据准备
- `prepare_data.py` - 数据准备脚本
  - 词表生成
  - 样本数据创建

- `generate_r2r_enhanced_data.py` - R2R 增强数据生成
  - 路径采样
  - 视觉特征生成
  - 候选方向生成

- `sample_r2r_paths.py` - R2R 路径采样

### 指令生成
- `generate_chinese_instructions.py` - 中文指令生成（百炼 API）
- `llm_bailian.py` - 阿里云百炼 API 封装
- `llm_bailian_anthropic.py` - Anthropic API 封装
- `batch_generate_instructions.py` - 批量指令生成
- `machine_translate.py` - 机器翻译（对照实验）

### 评估脚本
- `evaluate_vln_model.py` - 模型评估
  - SR/SPL 指标计算
  - Oracle SR 计算
  - DTW 计算
  - 详细结果输出

- `vln_evaluation.py` - 评估工具函数

### 可视化
- `generate_paper_figures.py` - 论文图表生成
  - 训练曲线对比
  - SR/SPL 对比图
  - 消融实验图
  - 置信度分布图
  - 架构图（TikZ）

- `visualize_attention.py` - 注意力可视化
  - 单样本注意力图
  - 按类型平均注意力

### 分析脚本
- `generate_qualitative_analysis.py` - 定性分析
  - 成功/失败案例分析
  - 指令类型分析
  - 误差模式分析

- `compare_mt_vs_llm.py` - 机器翻译 vs LLM 对比
- `evaluate_instructions.py` - 指令质量评估

## 辅助脚本

- `test_bailian.py` - 百炼 API 测试
- `prepare_r2r_english.py` - R2R 英文数据准备
- `process_r2r_real_data.py` - R2R 真实数据处理
- `vln_attention_example.py` - 注意力示例

## 使用推荐

### 快速开始
```bash
# 1. 准备数据
python code/prepare_data.py

# 2. 生成训练数据
python code/generate_r2r_enhanced_data.py

# 3. 训练模型
python code/train_vln_r2r_enhanced.py

# 4. 评估模型
python code/evaluate_vln_model.py

# 5. 可视化结果
python code/generate_paper_figures.py
```

### 核心流程

```
数据准备 → 指令生成 → 训练 → 评估 → 可视化
   ↓           ↓         ↓       ↓         ↓
prepare   generate  train  evaluate  figures
```

## 依赖关系

```
vln_baseline_model.py  ← 模型定义
    ↑
    | 被以下脚本导入
    |
train_vln_r2r_enhanced.py
evaluate_vln_model.py
visualize_attention.py
```

## 文件大小参考

| 文件 | 大小 | 说明 |
|------|------|------|
| vln_baseline_model.py | 12KB | 模型定义 |
| train_vln_r2r_enhanced.py | 14KB | 训练脚本 |
| evaluate_vln_model.py | 15KB | 评估脚本 |
| generate_r2r_enhanced_data.py | 18KB | 数据生成 |
| visualize_attention.py | 11KB | 注意力可视化 |
