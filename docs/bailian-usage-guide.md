# 阿里云百炼 LLM 使用指南

> 用于 VLN 中文指令生成和评估

---

## 一、快速开始

### 1.1 开通服务

1. 访问阿里云百炼控制台：https://bailian.console.aliyun.com/
2. 注册/登录阿里云账号
3. 开通百炼服务
4. 创建 API Key

### 1.2 设置 API Key

**方法 1：环境变量（推荐）**

```bash
export DASHSCOPE_API_KEY="sk-你的 api key"
```

**方法 2：代码中传入**

```python
from llm_bailian import BailianLLM

llm = BailianLLM(api_key="sk-你的 api key")
```

### 1.3 安装依赖

```bash
source vln-env/bin/activate
pip install dashscope
```

---

## 二、使用示例

### 2.1 基础对话

```python
from llm_bailian import BailianLLM

# 初始化
llm = BailianLLM(model="qwen-max")

# 调用
response = llm.chat("你好，请介绍一下你自己。")

if response.success:
    print(f"回复：{response.content}")
    print(f"Token 使用：{response.usage}")
else:
    print(f"调用失败：{response.error}")
```

### 2.2 VLN 指令生成

```python
from llm_bailian import VLNInstructionGenerator

# 初始化生成器
generator = VLNInstructionGenerator(model="qwen-max")

# 准备路径信息
path_info = {
    "path_id": "001",
    "scene_type": "住宅",
    "start_location": "客厅入口",
    "waypoints": ["沙发", "茶几", "拱门", "楼梯"],
    "end_location": "二楼卧室窗边",
    "distance": 15,
    "english_reference": "Walk past the sofa and go upstairs to the bedroom."
}

# 生成指令
instructions = generator.generate(path_info, num_variants=3)

for i, instr in enumerate(instructions, 1):
    print(f"{i}. {instr}")
```

**输出示例**：
```
1. 从客厅进来走过沙发，穿过拱门上楼梯，到二楼卧室窗边停下。
2. 进来后路过沙发和茶几，从拱门过去上楼梯，卧室窗户在那边。
3. 直走经过沙发，从前面拱门穿过去上楼，走到卧室窗口。
```

### 2.3 批量生成

```python
from llm_bailian import VLNInstructionGenerator

generator = VLNInstructionGenerator(model="qwen-turbo")  # turbo 更便宜

# 准备多个路径
path_list = [
    {
        "path_id": "001",
        "scene_type": "住宅",
        "start_location": "玄关",
        "waypoints": ["餐桌", "厨房", "后门"],
        "end_location": "后院",
        "distance": 10,
        "english_reference": "Go past the dining table and kitchen to the backyard."
    },
    {
        "path_id": "002",
        "scene_type": "办公室",
        "start_location": "电梯口",
        "waypoints": ["前台", "走廊", "会议室"],
        "end_location": "会议室 A",
        "distance": 20,
        "english_reference": "From elevator, pass reception and go down the hall to Meeting Room A."
    },
    # ... 更多路径
]

# 批量生成
results = generator.generate_batch(path_list)

# 保存结果
import json
with open("generated_instructions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### 2.4 指令评估

```python
from llm_bailian import VLNInstructionEvaluator

# 初始化评估器
evaluator = VLNInstructionEvaluator(model="qwen-max")

# 评估指令
instruction = "从客厅门口进来，走过沙发，穿过拱门走到走廊尽头。"
english_ref = "Enter from the living room door, walk past the sofa, and go through the archway to the end of the hallway."

result = evaluator.evaluate(instruction, english_ref)

print(f"自然度：{result['naturalness']}")
print(f"清晰度：{result['clarity']}")
print(f"可执行性：{result['executability']}")
print(f"完整性：{result['completeness']}")
print(f"综合分：{result['overall']}")
print(f"评语：{result['comments']}")
```

---

## 三、模型选择

| 模型 | 适用场景 | 价格 | 速度 |
|------|----------|------|------|
| **qwen-max** | 复杂任务、高质量要求 | 高 | 中 |
| **qwen-plus** | 平衡性能与成本 | 中 | 快 |
| **qwen-turbo** | 批量生成、快速迭代 | 低 | 很快 |

**推荐配置**：
- 指令生成：`qwen-turbo`（批量）、`qwen-plus`（高质量）
- 指令评估：`qwen-max`（需要准确评分）

---

## 四、成本估算

### 4.1 定价（参考）

| 模型 | 输入价格 | 输出价格 |
|------|----------|----------|
| qwen-max | ¥0.04/1K tokens | ¥0.12/1K tokens |
| qwen-plus | ¥0.004/1K tokens | ¥0.012/1K tokens |
| qwen-turbo | ¥0.002/1K tokens | ¥0.006/1K tokens |

### 4.2 生成 20k 指令成本估算

```
每条指令生成约消耗:
- 输入：~200 tokens (Prompt)
- 输出：~50 tokens (生成的指令)

使用 qwen-turbo:
20,000 × (200 × 0.002 + 50 × 0.006) / 1000 = ¥14

使用 qwen-plus:
20,000 × (200 × 0.004 + 50 × 0.012) / 1000 = ¥28
```

**结论**：批量生成建议使用 `qwen-turbo`，成本很低！

---

## 五、完整工作流

```python
"""
VLN 中文指令生成完整工作流
"""

from llm_bailian import VLNInstructionGenerator, VLNInstructionEvaluator
import json

# 1. 准备路径数据
with open("r2r_paths.json", "r") as f:
    path_list = json.load(f)

# 2. 生成指令（使用 turbo 节省成本）
print("开始生成指令...")
generator = VLNInstructionGenerator(model="qwen-turbo")
results = generator.generate_batch(path_list[:100])  # 先测试 100 条

# 3. 评估质量（使用 max 保证准确）
print("开始评估指令...")
evaluator = VLNInstructionEvaluator(model="qwen-max")

for item in results:
    eval_result = evaluator.evaluate(item["instruction"])
    item["evaluation"] = eval_result

# 4. 筛选高质量指令
high_quality = [
    item for item in results
    if item["evaluation"].get("overall", 0) >= 4.0
]

print(f"高质量指令：{len(high_quality)} / {len(results)}")

# 5. 保存结果
with open("chinese_vln_instructions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("完成！")
```

---

## 六、常见问题

### Q1: API 调用失败怎么办？

```python
# 检查 API Key
import os
print(os.environ.get("DASHSCOPE_API_KEY"))

# 捕获错误
response = llm.chat("...")
if not response.success:
    print(f"错误：{response.error}")
```

### Q2: 如何控制输出长度？

```python
# 在 Prompt 中明确要求
prompt = "请生成一条 20-30 字的中文导航指令..."

# 或修改 max_tokens 参数
response = llm.chat(prompt, max_tokens=100)  # 限制输出长度
```

### Q3: 如何提高生成质量？

1. **优化 Prompt**：提供更详细的路径信息
2. **增加示例**：Few-shot learning
3. **多次生成**：`num_variants=3` 然后选最好的
4. **后处理**：用规则过滤低质量指令

---

## 七、参考链接

- 阿里云百炼官网：https://bailian.console.aliyun.com/
- 通义千问 API 文档：https://help.aliyun.com/zh/dashscope/
- Python SDK: https://github.com/aliyun/alibabacloud-bailian-sdk

---

*文档创建：2026-03-17*
