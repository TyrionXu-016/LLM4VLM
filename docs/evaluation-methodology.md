# 中文 VLN 指令质量评估方法

> 评估 LLM 生成的导航指令质量的多维框架

---

## 一、评估维度总览

```
┌─────────────────────────────────────────────────────────┐
│              中文 VLN 指令质量评估体系                    │
├─────────────────────────────────────────────────────────┤
│  自动评估 (Automatic)    │  人工评估 (Human)            │
│  ├── 形式指标            │  ├── 自然度                  │
│  ├── 语言质量            │  ├── 清晰度                  │
│  └── 语义一致性          │  ├── 可执行性                │
│                          │  └── 信息完整性              │
├─────────────────────────────────────────────────────────┤
│  任务导向评估 (Task-Oriented)                           │
│  ├── 导航成功率 (SR)                                    │
│  └── 路径效率 (SPL)                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 二、自动评估指标

### 2.1 形式指标 (Formal Metrics)

| 指标 | 说明 | 合理范围 |
|------|------|----------|
| **字数** | 中文字符数量 | 20-60 字 |
| **动词密度** | 动作词占比 | 15-25% |
| **地标数量** | 物体/地点名词数 | 2-5 个 |
| **方向词数量** | 左/右/前/后等 | 2-4 个 |
| **句子数** | 分句数量 | 2-4 句 |

**计算示例**：
```
指令："从客厅门口进来，走过沙发和茶几，穿过前面的拱门走到走廊尽头。"

字数：32 字
动词：进来、走过、穿过、走 → 4 个 → 密度 12.5%
地标：客厅门口、沙发、茶几、拱门、走廊 → 5 个
方向词：进来、前面、尽头 → 3 个
句子数：2 句（逗号分隔）
```

---

### 2.2 语言质量指标

#### (1) 流畅度评分 (Perplexity-based)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(instruction, model_name="thu-coai/Chinese-LLaMA-2-7B"):
    """
    用中文语言模型计算困惑度，评估流畅度
    困惑度越低 → 语言越自然流畅
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(instruction, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    perplexity = torch.exp(outputs.loss).item()
    return perplexity
```

#### (2) 语法错误检测

```python
def check_grammar_errors(instruction):
    """
    检测常见语法错误
    """
    errors = []

    # 检查方向词搭配
    if "往左走" not in instruction and "向左走" not in instruction:
        if "往左" in instruction or "向左" in instruction:
            pass  # 可以接受
    else:
        pass

    # 检查不完整句子
    if instruction.endswith("的"):
        errors.append("句子可能不完整")

    # 检查重复用词
    words = instruction.split("，")
    if len(words) != len(set(words)):
        errors.append("可能存在重复表达")

    return errors
```

---

### 2.3 语义一致性指标

#### (1) 中英文语义对齐 (BERTScore)

```python
from bert_score import score

def calculate_bertscore(chinese, english):
    """
    计算中英文指令的语义相似度
    使用多语言 BERT 模型
    """
    P, R, F1 = score(
        [chinese],
        [english],
        lang="zh",
        model_type="bert-base-multilingual-cased",
        verbose=False
    )
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }
```

#### (2) 关键信息覆盖率

```python
def calculate_information_coverage(chinese, english):
    """
    检查中文指令是否覆盖了英文的关键信息
    """
    # 提取英文关键元素
    en_actions = ["walk", "turn", "go", "stop", "enter", "exit"]
    zh_actions = ["走", "转", "去", "停", "进入", "出去"]

    en_directions = ["left", "right", "straight", "forward"]
    zh_directions = ["左", "右", "直", "前"]

    # 检查覆盖
    action_covered = any(a in english.lower() for a in en_actions) and \
                     any(z in chinese for z in zh_actions)

    direction_covered = any(d in english.lower() for d in en_directions) and \
                        any(z in chinese for z in zh_directions)

    return {
        "action_coverage": action_covered,
        "direction_coverage": direction_covered,
        "overall_coverage": action_covered and direction_covered
    }
```

---

## 三、人工评估标准

### 3.1 评估量表

| 维度 | 1 分 | 2 分 | 3 分 | 4 分 | 5 分 |
|------|------|------|------|------|------|
| **自然度** | 机器翻译痕迹明显 | 有生硬表达 | 基本流畅 | 自然流畅 | 非常地道 |
| **清晰度** | 完全无法理解 | 多处歧义 | 部分模糊 | 清晰明确 | 非常清晰 |
| **可执行性** | 无法导航 | 多处错误 | 需猜测 | 可执行 | 唯一确定 |
| **信息完整性** | 缺失关键信息 | 信息不足 | 基本完整 | 信息充足 | 详尽完整 |

---

### 3.2 人工评估示例

**指令 1**：
> "从客厅门口进来，走过沙发和茶几，穿过前面的拱门走到走廊尽头，卫生间就到了。"

| 维度 | 评分 | 评语 |
|------|------|------|
| 自然度 | 5 | 表达地道，"就到了"很口语化 |
| 清晰度 | 5 | 每个步骤都很清楚 |
| 可执行性 | 5 | 可以唯一确定路径 |
| 完整性 | 4 | 缺少距离信息 |
| **平均分** | **4.75** | |

**指令 2**：
> "背对水槽，路过冰箱和储藏室，从后门出去就是洗衣房。"

| 维度 | 评分 | 评语 |
|------|------|------|
| 自然度 | 4 | "背对"稍显正式 |
| 清晰度 | 5 | 指令明确 |
| 可执行性 | 4 | "后门"可能有歧义 |
| 完整性 | 3 | 较短，缺少细节 |
| **平均分** | **4.0** | |

---

### 3.3 人工评估实施

#### 评估人员要求
- 母语中文使用者
- 有空间方向感
- 熟悉室内环境描述

#### 评估流程
```
1. 向评估员展示指令
2. 展示对应的路径（可选：可视化或视频）
3. 评估员填写评分表
4. 收集所有评估员的评分
5. 计算平均分和一致性 (Krippendorff's α)
```

#### 评估表模板

```markdown
## 指令评估表

**指令 ID**: _____

**指令内容**: _________________________________

### 评分（1-5 分）
- 自然度：□1 □2 □3 □4 □5
- 清晰度：□1 □2 □3 □4 □5
- 可执行性：□1 □2 □3 □4 □5
- 信息完整性：□1 □2 □3 □4 □5

### 备注
这条指令的问题或改进建议：
_________________________________
```

---

## 四、任务导向评估

### 4.1 导航成功率 (Success Rate)

**方法**：用生成的指令训练 VLN 模型，在验证集上测试

```python
def evaluate_navigation_success(model, instructions, ground_truth_paths):
    """
    评估指令的导航有效性

    Args:
        model: 训练好的 VLN 模型
        instructions: 中文指令列表
        ground_truth_paths: 真实路径

    Returns:
        success_rate: 成功率
        spl: 路径效率
    """
    successes = []
    spl_scores = []

    for instr, gt_path in zip(instructions, ground_truth_paths):
        # 模型导航
        predicted_path = model.navigate(instr)

        # 计算是否成功（距离 < 3m）
        distance = calculate_distance(predicted_path.end, gt_path.end)
        success = 1 if distance < 3.0 else 0
        successes.append(success)

        # 计算路径效率
        shortest_length = gt_path.length
        actual_length = predicted_path.length
        spl = success * (shortest_length / max(actual_length, 0.1))
        spl_scores.append(spl)

    return {
        "success_rate": sum(successes) / len(successes),
        "spl": sum(spl_scores) / len(spl_scores)
    }
```

---

### 4.2 人工导航验证

**方法**：让人类根据指令在仿真环境中导航

```
流程：
1. 评估员进入仿真环境
2. 阅读中文指令
3. 尝试导航到目标
4. 记录是否成功到达

指标：
- 人类成功率 (Human Success Rate)
- 平均导航时间
- 路径效率
```

---

## 五、综合评估框架

### 5.1 评估流程

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 自动筛选                                        │
│ - 字数检查 (20-60 字)                                    │
│ - 语法错误检测                                          │
│ - 关键信息覆盖                                          │
│ → 通过率约 80%                                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: LLM 自动评分                                     │
│ - 用 GPT-4/Claude 评估四个维度                           │
│ - 每个维度 1-5 分                                         │
│ → 获取批量评分                                           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: 人工抽检                                        │
│ - 随机抽取 100 条指令                                     │
│ - 3 名评估员独立评分                                      │
│ - 计算评分者一致性                                        │
│ → 验证自动评分可靠性                                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: 任务验证                                        │
│ - 选取 Top-50 指令训练模型                                │
│ - 在验证集测试 SR/SPL                                     │
│ → 验证任务有效性                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 5.2 综合评分公式

```
综合分数 = 0.3 × 自动评分 + 0.4 × 人工评分 + 0.3 × 任务评分

其中：
- 自动评分 = (语言质量 + 语义一致性) / 2
- 人工评分 = (自然度 + 清晰度 + 可执行性 + 完整性) / 4
- 任务评分 = 导航成功率 (SR)
```

---

## 六、Python 实现

### 6.1 完整评估类

```python
import json
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class EvaluationResult:
    path_id: str
    instruction: str

    # 自动评估
    word_count: int
    perplexity: float
    bertscore_f1: float

    # LLM 评估
    llm_naturalness: float
    llm_clarity: float
    llm_executability: float
    llm_completeness: float
    llm_overall: float

    # 综合
    final_score: float
    quality_level: str  # "优秀", "良好", "合格", "不合格"


class ChineseVLNEvaluator:
    def __init__(self, llm_api_key=None):
        self.llm_api_key = llm_api_key

    def evaluate_single(self, instruction, english_ref=None) -> EvaluationResult:
        """评估单条指令"""
        result = EvaluationResult(
            path_id="unknown",
            instruction=instruction,
            word_count=len(instruction),
            perplexity=self._calculate_perplexity(instruction),
            bertscore_f1=self._calculate_bertscore(instruction, english_ref) if english_ref else 0.0,
            llm_naturalness=self._llm_rate(instruction, "自然度"),
            llm_clarity=self._llm_rate(instruction, "清晰度"),
            llm_executability=self._llm_rate(instruction, "可执行性"),
            llm_completeness=self._llm_rate(instruction, "完整性"),
            llm_overall=0.0,
            final_score=0.0,
            quality_level="待评定"
        )

        # 计算 LLM 综合分
        result.llm_overall = (
            result.llm_naturalness +
            result.llm_clarity +
            result.llm_executability +
            result.llm_completeness
        ) / 4

        # 计算最终分数（简化版）
        result.final_score = result.llm_overall * 0.7 + (result.bertscore_f1 * 10) * 0.3
        result.quality_level = self._score_to_level(result.final_score)

        return result

    def _llm_rate(self, instruction, dimension) -> float:
        """用 LLM 评估某个维度"""
        prompt = f"""
请评估以下中文导航指令的{dimension}（1-5 分）：

指令：{instruction}

评分标准：
1 分 = 很差
2 分 = 较差
3 分 = 一般
4 分 = 良好
5 分 = 优秀

请只输出分数（1-5 的数字）：
"""
        # 调用 LLM API 获取评分
        # response = call_llm_api(prompt)
        # return float(response)
        return 4.0  # 占位

    def _calculate_perplexity(self, instruction) -> float:
        """计算困惑度"""
        # 实现略
        return 0.0

    def _calculate_bertscore(self, zh, en) -> float:
        """计算 BERTScore"""
        # 实现略
        return 0.0

    def _score_to_level(self, score) -> str:
        if score >= 4.5:
            return "优秀"
        elif score >= 3.5:
            return "良好"
        elif score >= 2.5:
            return "合格"
        else:
            return "不合格"

    def batch_evaluate(self, instructions: List[Dict]) -> List[EvaluationResult]:
        """批量评估"""
        results = []
        for item in instructions:
            result = self.evaluate_single(
                item["instruction"],
                item.get("english_reference")
            )
            result.path_id = item.get("path_id", "unknown")
            results.append(result)
        return results
```

---

## 七、评估报告模板

```markdown
# 中文 VLN 指令质量评估报告

## 执行摘要
- 评估指令总数：XXX 条
- 平均综合分数：X.X / 5.0
- 质量分布：优秀 XX%, 良好 XX%, 合格 XX%, 不合格 XX%

## 自动评估结果
| 指标 | 平均值 | 标准差 |
|------|--------|--------|
| 字数 | XX.X | X.X |
| 困惑度 | XX.X | X.X |
| BERTScore | 0.XX | 0.XX |

## LLM 评估结果
| 维度 | 平均分 |
|------|--------|
| 自然度 | X.X |
| 清晰度 | X.X |
| 可执行性 | X.X |
| 完整性 | X.X |

## 人工评估结果（抽样 100 条）
| 维度 | 平均分 | 评分者一致性 (α) |
|------|--------|------------------|
| 自然度 | X.X | 0.XX |
| 清晰度 | X.X | 0.XX |
| 可执行性 | X.X | 0.XX |
| 完整性 | X.X | 0.XX |

## 问题分类统计
| 问题类型 | 出现次数 | 占比 |
|----------|----------|------|
| 字数过短 | XX | XX% |
| 缺少地标 | XX | XX% |
| 方向模糊 | XX | XX% |
| 翻译腔 | XX | XX% |

## 建议
1. ...
2. ...
3. ...
```

---

*文档创建：2026-03-17*
