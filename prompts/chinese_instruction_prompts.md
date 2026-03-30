# VLN 中文指令生成 Prompt 模板

> 用于批量生成中文视觉语言导航指令

---

## 模板 1：基于路径描述生成

```python
PROMPT_PATH_TO_INSTRUCTION = """
你是一个专业的导航指令标注员。请根据给定的路径信息，生成一条自然的中文导航指令。

## 路径信息
- 场景类型：{scene_type}
- 起点：{start_location}
- 途经点：{waypoints}
- 终点：{end_location}
- 路径长度：约 {distance} 米
- 动作数量：{num_actions} 步

## 指令要求
1. 长度：30-60 字之间
2. 必须包含：
   - 方向信息（左转、右转、直走等）
   - 地标参考（沙发、桌子、楼梯等）
   - 距离提示（几步、走到头、穿过等）
3. 语言风格：
   - 自然流畅，像日常说话
   - 避免机械化的"先...然后...最后"结构
   - 可以适当使用口语化表达

## 示例
英文原句："Walk past the sofa, turn left at the painting, and stop by the window."
优质中文："走过沙发，看到画后左转，走到窗边停下。"

## 你的输出
"""
```

---

## 模板 2：英文翻译 + 多样化

```python
PROMPT_TRANSLATE_VARIANTS = """
请将以下英文导航指令翻译成中文，并生成 3 个不同风格的变体。

## 英文原句
"{english_instruction}"

## 路径参考（帮助理解）
- 起点：{start}
- 终点：{end}
- 关键地标：{landmarks}

## 输出格式
### 变体 1（简洁风格）
[简洁直接的表达]

### 变体 2（详细风格）
[详细描述每个步骤]

### 变体 3（地标风格）
[重点强调视觉地标]

## 翻译原则
1. 准确传达原意
2. 符合中文空间表达习惯
3. 自然流畅，避免翻译腔
"""
```

---

## 模板 3：Few-shot 风格学习

```python
PROMPT_FEW_SHOT = """
学习以下示例的风格，为新的路径生成中文导航指令。

## 示例 1
路径：客厅入口 → 沙发 → 拱门 → 二楼卧室
指令："从门口进来，走过沙发，穿过前面的拱门，上楼梯到卧室。"

## 示例 2
路径：玄关 → 走廊 → 厨房 → 阳台
指令："进门后直走，穿过走廊，路过厨房，走到阳台停下。"

## 示例 3
路径：书桌 → 窗户 → 门 → 楼梯
指令："从书桌那边往窗户走，然后转向门的方向，下楼。"

## 新任务
路径：{path_description}

请生成指令：
"""
```

---

## 模板 4：批量生成（JSON 格式）

```python
PROMPT_BATCH_GENERATION = """
请为以下路径批量生成中文导航指令。

## 输入路径列表
{path_list}

## 输出格式（JSON）
```json
[
  {{
    "path_id": "路径 ID",
    "instruction": "生成的中文指令",
    "style": "风格标签",
    "word_count": 字数
  }}
]
```

## 要求
1. 每条指令 30-60 字
2. 风格多样化
3. 避免重复表达
"""
```

---

## 模板 5：质量评估

```python
PROMPT_QUALITY_EVALUATION = """
请评估以下中文导航指令的质量。

## 指令内容
"{instruction}"

## 评估维度（1-5 分）
1. 自然度：是否像人类说的话？
2. 清晰度：指令是否明确无歧义？
3. 可执行性：能否唯一确定导航路径？
4. 信息完整性：是否包含足够的导航信息？

## 输出格式
```json
{{
  "naturalness": 分数,
  "clarity": 分数,
  "executability": 分数,
  "completeness": 分数,
  "overall": 平均分,
  "comments": "具体评价和改进建议"
}}
```
"""
```

---

## 使用示例

```python
import openai

def generate_chinese_instruction(path_info, template="path_to_instruction"):
    if template == "path_to_instruction":
        prompt = PROMPT_PATH_TO_INSTRUCTION.format(**path_info)
    elif template == "translate_variants":
        prompt = PROMPT_TRANSLATE_VARIANTS.format(**path_info)
    elif template == "few_shot":
        prompt = PROMPT_FEW_SHOT.format(**path_info)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )

    return response.choices[0].message.content

# 使用示例
path_info = {
    "scene_type": "住宅",
    "start_location": "客厅入口",
    "waypoints": ["沙发", "茶几", "拱门", "楼梯"],
    "end_location": "二楼卧室窗边",
    "distance": 15,
    "num_actions": 5
}

result = generate_chinese_instruction(path_info)
print(result)
```

---

## 预期输出示例

**输入**：
```
场景类型：住宅
起点：客厅入口
途经点：沙发、茶几、拱门、楼梯
终点：二楼卧室窗边
```

**可能输出**：
> "从客厅门口进来，往前走过沙发和茶几，看到拱门后穿过去。然后上楼梯到二楼，走到卧室的窗户那边停下。"

---

*创建时间：2026-03-17*
