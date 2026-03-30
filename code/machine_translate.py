"""
使用机器翻译将 R2R 英文指令翻译成中文

对比：机器翻译 vs LLM 直接生成
"""

import sys
import json

sys.path.insert(0, '/Users/tyrion/Projects/Papers/code')

from dotenv import load_dotenv
load_dotenv()

from llm_bailian import BailianLLM, Config


def translate_instruction(english_text: str, llm: BailianLLM) -> str:
    """
    使用 LLM 翻译英文指令为中文

    Args:
        english_text: 英文指令
        llm: LLM 实例

    Returns:
        中文翻译
    """
    prompt = f"""请将以下英文导航指令翻译成自然流畅的中文：

英文：{english_text}

要求：
1. 保持原意准确
2. 使用自然的中文表达
3. 保留地标和方向信息
4. 符合口语习惯

中文翻译："""

    response = llm.chat(prompt, max_tokens=100)
    if response.success:
        return response.content.strip()
    else:
        print(f"翻译失败：{response.error}")
        return ""


def batch_translate(input_file: str, output_file: str, model: str = "qwen3.5-plus"):
    """
    批量翻译英文指令

    Args:
        input_file: 输入英文指令文件
        output_file: 输出翻译文件
        model: 使用的模型
    """
    # 加载英文指令
    with open(input_file, 'r', encoding='utf-8') as f:
        instructions = json.load(f)

    print(f"加载了 {len(instructions)} 条英文指令")
    print(f"翻译模型：{model}")
    print()

    # 创建翻译 LLM
    llm = BailianLLM(model=model)

    # 批量翻译
    translated = []
    for i, item in enumerate(instructions, 1):
        english = item['english_reference']
        print(f"[{i}/{len(instructions)}] 翻译 {item['path_id']}...")

        chinese = translate_instruction(english, llm)

        translated.append({
            "path_id": item['path_id'],
            "scene_type": item['scene_type'],
            "english_reference": english,
            "machine_translation": chinese,
            "start_location": item['start_location'],
            "waypoints": item['waypoints'],
            "end_location": item['end_location'],
            "distance": item['distance'],
        })

        print(f"  英文：{english[:50]}...")
        print(f"  翻译：{chinese}")
        print()

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    print(f"翻译完成！已保存到：{output_file}")
    return translated


if __name__ == "__main__":
    translated = batch_translate(
        input_file="/Users/tyrion/Projects/Papers/data/r2r_english_samples.json",
        output_file="/Users/tyrion/Projects/Papers/data/r2r_machine_translations.json",
        model="qwen3.5-plus"
    )
