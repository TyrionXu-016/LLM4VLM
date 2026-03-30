"""
VLN 中文指令生成示例
使用模拟的 LLM API 生成中文导航指令
"""

# 模拟 R2R 风格的路径数据
PATH_SAMPLES = [
    {
        "path_id": "1",
        "scene_type": "住宅",
        "start_location": "客厅入口",
        "waypoints": ["沙发", "茶几", "拱门", "走廊"],
        "end_location": "走廊尽头的卫生间",
        "distance": 12,
        "num_actions": 4,
        "english_reference": "Walk past the sofa and coffee table, go through the archway into the hallway. Stop at the bathroom at the end."
    },
    {
        "path_id": "2",
        "scene_type": "住宅",
        "start_location": "玄关",
        "waypoints": ["餐桌", "厨房岛台", "滑动门"],
        "end_location": "后院露台",
        "distance": 8,
        "num_actions": 3,
        "english_reference": "Enter and turn right past the dining table. Walk past the kitchen island and go through the sliding door to the patio."
    },
    {
        "path_id": "3",
        "scene_type": "办公室",
        "start_location": "接待处",
        "waypoints": ["前台", "玻璃门", "会议室"],
        "end_location": "会议室最里面的座位",
        "distance": 15,
        "num_actions": 5,
        "english_reference": "From reception, walk straight past the front desk. Go through the glass doors and turn left into the conference room. Take a seat at the far end."
    },
    {
        "path_id": "4",
        "scene_type": "住宅",
        "start_location": "楼梯底部",
        "waypoints": ["二楼扶手", "挂画", "地毯"],
        "end_location": "主卧衣柜前",
        "distance": 10,
        "num_actions": 4,
        "english_reference": "Go upstairs and turn right. Walk past the painting on the wall and step onto the rug. Stop in front of the bedroom closet."
    },
    {
        "path_id": "5",
        "scene_type": "住宅",
        "start_location": "厨房水槽",
        "waypoints": ["冰箱", "储藏室", "后门"],
        "end_location": "洗衣房",
        "distance": 6,
        "num_actions": 3,
        "english_reference": "Turn away from the sink and walk past the refrigerator. Go past the pantry and exit through the back door into the laundry room."
    },
]

# 提示模板
INSTRUCTION_PROMPT = """
你是一个专业的导航指令标注员。请根据给定的路径信息，生成一条自然的中文导航指令。

## 路径信息
- 场景类型：{scene_type}
- 起点：{start_location}
- 途经点：{waypoints}
- 终点：{end_location}
- 路径长度：约 {distance} 米

## 指令要求
1. 长度：20-50 字之间
2. 必须包含：方向信息、地标参考、距离提示
3. 语言风格：自然流畅，像日常说话

## 你的输出
"""

def generate_instruction_llm(path_info):
    """
    调用 LLM 生成中文指令

    注意：实际使用时需要配置 API key
    这里提供两种方案：
    1. 调用真实 API（GPT-4, Claude, 文心一言等）
    2. 本地运行开源模型（ChatGLM, Qwen 等）
    """
    prompt = INSTRUCTION_PROMPT.format(
        scene_type=path_info["scene_type"],
        start_location=path_info["start_location"],
        waypoints="、".join(path_info["waypoints"]),
        end_location=path_info["end_location"],
        distance=path_info["distance"]
    )

    # === 方案 1: 调用 API（需要配置）===
    # from openai import OpenAI
    # client = OpenAI(api_key="your-api-key")
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content

    # === 方案 2: 模拟输出（用于测试）===
    return simulate_llm_response(path_info)


def simulate_llm_response(path_info):
    """
    模拟 LLM 输出（用于没有 API 时的测试）
    """
    responses = {
        "1": "从客厅门口进来，走过沙发和茶几，穿过前面的拱门走到走廊尽头，卫生间就到了。",
        "2": "进门后右拐经过餐桌，路过厨房岛台，从滑动门出去就是后院露台。",
        "3": "从接待处直走经过前台，穿过玻璃门后左转进入会议室，走到最里面坐下。",
        "4": "上楼梯后右转，沿着墙上的画直走，踩到地毯后继续走，主卧衣柜就在前面。",
        "5": "背对水槽，路过冰箱和储藏室，从后门出去就是洗衣房。"
    }
    return responses.get(path_info["path_id"], "生成失败")


def generate_variants(instruction, num_variants=3):
    """
    生成同一路径的多个指令变体
    """
    variants_prompt = f"""
请将以下中文导航指令改写成 {num_variants} 个不同风格的版本。

原句：{instruction}

要求：
1. 保持语义不变
2. 每句长度 20-50 字
3. 风格差异明显（简洁、详细、口语化等）
"""
    # 实际使用时调用 API
    return []


def batch_generate(paths):
    """
    批量生成中文指令
    """
    results = []
    for path in paths:
        print(f"正在生成路径 {path['path_id']} 的指令...")
        instruction = generate_instruction_llm(path)
        results.append({
            "path_id": path["path_id"],
            "instruction": instruction,
            "english_reference": path["english_reference"]
        })
        print(f"  ✓ 完成：{instruction}")
    return results


# === 主程序 ===
if __name__ == "__main__":
    print("=" * 60)
    print("VLN 中文指令生成示例")
    print("=" * 60)
    print()

    # 批量生成
    results = batch_generate(PATH_SAMPLES)

    print()
    print("=" * 60)
    print("生成结果汇总")
    print("=" * 60)
    print()

    for r in results:
        print(f"路径 {r['path_id']}:")
        print(f"  中文：{r['instruction']}")
        print(f"  英文：{r['english_reference']}")
        print()

    # 保存结果
    import json
    output_file = "/Users/tyrion/Projects/Papers/data/generated_instructions_sample.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到：{output_file}")
