"""
批量生成中文 VLN 指令

使用阿里云百炼 API 进行指令生成
"""

import sys
import json
import time

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dotenv import load_dotenv
load_dotenv()

from llm_bailian import VLNInstructionGenerator, Config


def batch_generate(input_file: str, output_file: str, model: str = None):
    """
    批量生成指令

    Args:
        input_file: 输入路径文件
        output_file: 输出指令文件
        model: 使用的模型
    """
    # 加载路径数据
    with open(input_file, 'r', encoding='utf-8') as f:
        paths = json.load(f)

    print(f"加载了 {len(paths)} 条路径")
    print(f"生成模型：{model or Config.GENERATION_MODEL}")
    print(f"每个路径生成变体数：{paths[0].get('num_variants', 3) if paths else 3}")
    print()

    # 创建生成器
    generator = VLNInstructionGenerator(model=model)

    # 批量生成
    start_time = time.time()
    results = generator.generate_batch(paths)
    elapsed = time.time() - start_time

    print()
    print(f"生成完成！")
    print(f"  总指令数：{len(results)}")
    print(f"  耗时：{elapsed:.1f} 秒")
    print(f"  平均速度：{len(results)/elapsed:.1f} 条/秒")

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  已保存到：{output_file}")

    return results


if __name__ == "__main__":
    results = batch_generate(
        input_file=str(REPO_ROOT / "data" / "sample_paths.json"),
        output_file=str(REPO_ROOT / "data" / "generated_instructions.json"),
        model="qwen3.5-plus"  # 使用你配置的模型
    )

    # 显示部分结果
    print()
    print("=" * 70)
    print("生成结果示例:")
    print("=" * 70)
    for item in results[:6]:
        print(f"\n【{item['path_id']}】变体 {item['variant']}")
        print(f"指令：{item['instruction']}")
