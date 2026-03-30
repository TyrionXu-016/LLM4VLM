# VLN 研究入门指南

> 针对中文 VLN + 跨语言迁移方向的新手入门路径

---

## 第一周：文献阅读

### Day 1-2: VLN 基础

**必读论文 1**: Anderson et al. (2018) - Vision-and-Language Navigation

**阅读重点**：
- [ ] VLN 任务的形式化定义
- [ ] R2R 数据集的构建方式
- [ ] Speaker-Follower 模型架构
- [ ] 评估指标（SR, SPL, Oracle）

**笔记模板**：
```
论文标题：
核心问题：
方法概述：
主要贡献：
我的思考：
```

---

### Day 3-4: 预训练方法

**必读论文 2**: PREVALENT (CVPR 2020)

**阅读重点**：
- [ ] 预训练任务设计
- [ ] 自监督数据增强
- [ ] 为什么简单方法有效

**必读论文 3**: VLN-BERT (CVPR 2021)

**阅读重点**：
- [ ] 历史状态编码
- [ ] 多模态融合方式
- [ ] 性能提升来源

---

### Day 5-7: 最新进展

**选读**（根据兴趣选择 2-3 篇）：
- [ ] HAMT (ICCV 2021) - History-Aware Transformer
- [ ] MapGPT (CVPR 2024) - 认知地图 + LLM
- [ ] ThinkNav (2023) - 思维链导航
- [ ] LLM4VN (2023) - 零样本 VLN

---

## 第二周：代码实践

### Day 1-2: 环境验证

```bash
# 激活环境
source vln-env/bin/activate

# 验证 PyTorch
python -c "import torch; print(torch.__version__)"

# 验证 MPS 加速
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

---

### Day 3-4: 模型代码阅读

**推荐代码库**: VLNCE

```bash
# 克隆代码
git clone https://github.com/facebookresearch/vln-ce.git
cd vln-ce

# 阅读关键文件
# 1. 模型定义
# 2. 数据加载
# 3. 训练循环
```

**阅读任务**：
- [ ] 理解数据加载流程
- [ ] 理解模型前向传播
- [ ] 理解损失函数

---

### Day 5-7: 小规模实验

**目标**：在验证集上跑通基线

```python
# 示例代码结构
import torch
from transformers import BertModel

class VLNBaseline:
    def __init__(self):
        self.vision_encoder = ...
        self.language_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.action_head = ...

    def forward(self, images, instructions):
        # 实现前向传播
        pass
```

---

## 第三周及以后：研究推进

### 方向选择

根据兴趣选择一个具体方向深入：

**A. 数据方向**
- 中文指令标注规范设计
- 标注工具开发
- 质量评估方法

**B. 方法方向**
- 跨语言适配器设计
- 对比学习策略
- 零样本迁移方法

**C. 分析方向**
- 中英空间语言对比
- 错误类型分析
- 模型可解释性

---

## 附录：常用资源

### 数据集
- R2R: https://github.com/peteanderson80/Matterport3DSimulator
- RxR: https://github.com/google-talkwalk/RxR-AGENT

### 代码库
- VLNCE: https://github.com/facebookresearch/vln-ce
- VLN-BERT: https://github.com/cshizhe/VLN-BERT
- HAMT: https://github.com/cshizhe/HAMT

### 阅读工具
- Zotero: 文献管理
- Notion/Obsidian: 笔记整理
- Connected Papers: 论文关系图

---

*创建时间：2026-03-17*
