"""
测试阿里云百炼 LLM 封装 (Anthropic 兼容接口)

运行前请确保：
1. 已安装依赖：pip install anthropic python-dotenv
2. 已设置 API Key: 复制 .env.example 为 .env 并填入 API Key
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../LLM4VLM
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# 先加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
from llm_bailian import Config, BailianLLM, VLNInstructionGenerator, VLNInstructionEvaluator

# 检查 API Key
api_key = Config.DASHSCOPE_API_KEY

if not api_key or api_key == "sk-your-api-key-here":
    print("=" * 60)
    print("⚠️  未检测到有效的 API Key")
    print("=" * 60)
    print()
    print("请设置 API Key:")
    print()
    print("  方法 1: .env 文件（推荐）")
    print('  复制 .env.example 为 .env 并填入你的 API Key')
    print()
    print("  方法 2: 环境变量")
    print('  export DASHSCOPE_API_KEY="sk-你的 key"')
    print()
    print("获取 API Key:")
    print("  1. 访问：https://bailian.console.aliyun.com/")
    print("  2. 登录阿里云账号")
    print("  3. 开通百炼服务")
    print("  4. 创建 API Key")
    print()
    sys.exit(1)

print(f"✓ API Key 已加载：{api_key[:15]}...")
print(f"API Base: {Config.API_BASE_URL}")
print()

# ============================================================
# 测试 1: 基础对话
# ============================================================
print("=" * 60)
print("测试 1: 基础对话")
print("=" * 60)

llm = BailianLLM(model="qwen3.5-plus")
response = llm.chat("你好，请用一句话介绍你自己。", max_tokens=50)

if response.success:
    print(f"✓ 调用成功")
    print(f"回复：{response.content}")
    print(f"Token 使用：{response.usage}")
else:
    print(f"✗ 调用失败：{response.error}")

print()

# ============================================================
# 测试 2: VLN 指令生成
# ============================================================
print("=" * 60)
print("测试 2: VLN 指令生成")
print("=" * 60)

path_info = {
    "path_id": "test_001",
    "scene_type": "住宅",
    "start_location": "客厅入口",
    "waypoints": ["沙发", "茶几", "拱门", "楼梯"],
    "end_location": "二楼卧室窗边",
    "distance": 15,
    "english_reference": "Walk past the sofa and go upstairs to the bedroom window."
}

generator = VLNInstructionGenerator(model="qwen3.5-plus")
instructions = generator.generate(path_info, num_variants=3)

if instructions:
    print(f"✓ 生成了 {len(instructions)} 条指令:")
    for i, instr in enumerate(instructions, 1):
        print(f"  {i}. {instr}")
else:
    print("✗ 生成失败")

print()

# ============================================================
# 测试 3: VLN 指令评估
# ============================================================
print("=" * 60)
print("测试 3: VLN 指令评估")
print("=" * 60)

test_instruction = "从客厅门口进来，走过沙发和茶几，穿过前面的拱门走到走廊尽头。"

evaluator = VLNInstructionEvaluator(model="qwen3.5-plus")
result = evaluator.evaluate(test_instruction)

print(f"指令：{test_instruction}")
print()
print(f"评估结果:")
print(f"  自然度：{result.get('naturalness', 'N/A')}")
print(f"  清晰度：{result.get('clarity', 'N/A')}")
print(f"  可执行性：{result.get('executability', 'N/A')}")
print(f"  完整性：{result.get('completeness', 'N/A')}")
print(f"  综合分：{result.get('overall', 'N/A')}")
print(f"  评语：{result.get('comments', 'N/A')}")

print()
print("=" * 60)
print("测试完成!")
print("=" * 60)
