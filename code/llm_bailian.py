"""
阿里云百炼 LLM API 封装 - Anthropic 兼容接口

用于 VLN 中文指令生成和评估

使用步骤:
1. 开通阿里云百炼服务：https://bailian.console.aliyun.com/
2. 获取 API Key
3. 复制 .env.example 为 .env 并填入 API Key
4. 安装依赖：pip install anthropic python-dotenv

文档：https://help.aliyun.com/zh/dashscope/
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# 尝试从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import anthropic


@dataclass
class LLMResponse:
    """LLM 响应数据结构"""
    success: bool
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


class Config:
    """
    配置管理类，从环境变量或 .env 文件读取配置
    """
    # API Key
    DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

    # API Base URL (Anthropic 兼容接口)
    API_BASE_URL = os.environ.get("API_BASE_URL", "https://coding.dashscope.aliyuncs.com/apps/anthropic")

    # 默认模型
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "qwen3.5-plus")

    # 生成模型（批量生成用）
    GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "kimi-k2.5")

    # 评估模型
    EVALUATION_MODEL = os.environ.get("EVALUATION_MODEL", "qwen3-max-2026-01-23")

    # 温度参数
    TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

    # 最大生成长度
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "500"))

    # 指令长度限制
    MIN_INSTRUCTION_LENGTH = int(os.environ.get("MIN_INSTRUCTION_LENGTH", "20"))
    MAX_INSTRUCTION_LENGTH = int(os.environ.get("MAX_INSTRUCTION_LENGTH", "60"))

    # 生成变体数量
    DEFAULT_NUM_VARIANTS = int(os.environ.get("DEFAULT_NUM_VARIANTS", "3"))


class BailianLLM:
    """
    阿里云百炼 LLM 封装类 (Anthropic 兼容接口)

    支持的模型:
    - qwen3.5-plus: 平衡性能和成本
    - kimi-k2.5: Moonshot 模型
    - qwen3-max: 最强性能
    """

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        初始化 LLM

        Args:
            api_key: 阿里云 API Key，如不传则从环境变量/.env 文件读取
            model: 使用的模型名称，如不传则使用 Config.DEFAULT_MODEL
        """
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = Config.DASHSCOPE_API_KEY

        if not self.api_key:
            raise ValueError(
                "请设置 API Key: \n"
                "1. 复制 .env.example 为 .env 并填入 API Key\n"
                "2. 设置环境变量：export DASHSCOPE_API_KEY='sk-xxx'\n"
                "3. 或在初始化时传入：BailianLLM(api_key='sk-xxx')"
            )

        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=Config.API_BASE_URL
        )
        self.model = model if model else Config.DEFAULT_MODEL

    def chat(self, prompt: str, system_message: Optional[str] = None,
             temperature: float = None, max_tokens: int = None) -> LLMResponse:
        """
        单次对话调用

        Args:
            prompt: 用户输入
            system_message: 系统提示
            temperature: 温度参数 (0.1-1.0)
            max_tokens: 最大生成长度

        Returns:
            LLMResponse 对象
        """
        if temperature is None:
            temperature = Config.TEMPERATURE
        if max_tokens is None:
            max_tokens = Config.MAX_TOKENS

        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            if system_message:
                kwargs["system"] = system_message

            response = self.client.messages.create(**kwargs)

            # 提取文本内容（处理 ThinkingBlock 和 TextBlock）
            content_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content_text += block.text

            return LLMResponse(
                success=True,
                content=content_text,
                model=self.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                error=None
            )

        except Exception as e:
            return LLMResponse(
                success=False,
                content="",
                model=self.model,
                usage={},
                error=str(e)
            )

    def chat_with_history(self, messages: List[Dict[str, str]],
                          temperature: float = None,
                          max_tokens: int = None) -> LLMResponse:
        """
        带历史记录的对话

        Args:
            messages: 对话历史
            temperature: 温度参数
            max_tokens: 最大生成长度

        Returns:
            LLMResponse 对象
        """
        if temperature is None:
            temperature = Config.TEMPERATURE
        if max_tokens is None:
            max_tokens = Config.MAX_TOKENS

        try:
            # 转换消息格式为 Anthropic 格式
            system_message = None
            anthropic_messages = []

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    system_message = content
                else:
                    anthropic_messages.append({"role": role, "content": content})

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": anthropic_messages
            }
            if system_message:
                kwargs["system"] = system_message

            response = self.client.messages.create(**kwargs)

            content_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content_text += block.text

            return LLMResponse(
                success=True,
                content=content_text,
                model=self.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                error=None
            )

        except Exception as e:
            return LLMResponse(
                success=False,
                content="",
                model=self.model,
                usage={},
                error=str(e)
            )


# ============================================================
# VLN 指令生成专用类
# ============================================================

class VLNInstructionGenerator:
    """
    VLN 中文指令生成器

    基于阿里云百炼 Qwen 模型
    """

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        初始化 VLN 指令生成器

        Args:
            api_key: 阿里云 API Key，如不传则从环境变量/.env 文件读取
            model: 使用的模型名称，如不传则使用 Config.GENERATION_MODEL
        """
        if model is None:
            model = Config.GENERATION_MODEL
        self.llm = BailianLLM(api_key=api_key, model=model)

        self.system_prompt = """
你是一个专业的导航指令标注员，擅长生成自然流畅的中文导航指令。

你的任务是根据给定的路径信息，生成符合以下要求的中文导航指令：

1. 长度要求：20-50 字
2. 必须包含：
   - 方向信息（左转、右转、直走等）
   - 地标参考（沙发、桌子、楼梯等）
   - 距离提示（几步、走到头、穿过等）
3. 语言风格：
   - 自然流畅，像日常说话
   - 避免机械化的"先...然后...最后"结构
   - 可以适当使用口语化表达（如"就到了"、"那边"）

请只输出生成的指令，不要输出其他内容。
"""

    def generate(self, path_info: Dict[str, Any], num_variants: int = 1) -> List[str]:
        """
        生成导航指令

        Args:
            path_info: 路径信息字典
            num_variants: 生成变体数量

        Returns:
            生成的指令列表
        """
        prompt = self._build_prompt(path_info)

        if num_variants == 1:
            response = self.llm.chat(prompt, system_message=self.system_prompt)
            if response.success:
                return [response.content.strip()]
            else:
                print(f"生成失败：{response.error}")
                return []
        else:
            prompt_with_variants = f"""
{prompt}

请生成 {num_variants} 个不同风格的版本，每个版本一行。
"""
            response = self.llm.chat(prompt_with_variants, system_message=self.system_prompt)
            if response.success:
                lines = response.content.strip().split("\n")
                return [line.strip() for line in lines if line.strip()]
            else:
                print(f"生成失败：{response.error}")
                return []

    def _build_prompt(self, path_info: Dict[str, Any]) -> str:
        """构建生成 Prompt"""
        waypoints = path_info.get("waypoints", [])
        if isinstance(waypoints, list):
            waypoints_str = "、".join(waypoints)
        else:
            waypoints_str = str(waypoints)

        prompt = f"""
## 路径信息
- 场景类型：{path_info.get('scene_type', '室内')}
- 起点：{path_info.get('start_location', '入口处')}
- 途经点：{waypoints_str}
- 终点：{path_info.get('end_location', '目标位置')}
- 路径长度：约{path_info.get('distance', 10)}米

## 你的任务
请为上述路径生成一条中文导航指令。
"""
        return prompt

    def generate_batch(self, path_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量生成指令

        Args:
            path_list: 路径信息列表

        Returns:
            包含 path_id 和 instruction 的结果列表
        """
        results = []

        for path_info in path_list:
            print(f"正在生成路径 {path_info.get('path_id', '?')} 的指令...")

            instructions = self.generate(
                path_info,
                num_variants=path_info.get("num_variants", 1)
            )

            for i, instr in enumerate(instructions):
                results.append({
                    "path_id": path_info.get("path_id", f"unknown_{len(results)}"),
                    "instruction": instr,
                    "variant": i + 1,
                    "english_reference": path_info.get("english_reference", ""),
                    "metadata": {
                        "scene_type": path_info.get("scene_type", ""),
                        "distance": path_info.get("distance", 0)
                    }
                })

            print(f"  ✓ 完成：{instructions[0] if instructions else '生成失败'}")

        return results


# ============================================================
# VLN 指令评估专用类
# ============================================================

class VLNInstructionEvaluator:
    """
    VLN 中文指令评估器

    基于阿里云百炼 Qwen 模型进行质量评分
    """

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        初始化 VLN 指令评估器

        Args:
            api_key: 阿里云 API Key，如不传则从环境变量/.env 文件读取
            model: 使用的模型名称，如不传则使用 Config.EVALUATION_MODEL
        """
        if model is None:
            model = Config.EVALUATION_MODEL
        self.llm = BailianLLM(api_key=api_key, model=model)

        self.eval_system_prompt = """
你是一个专业的语言评估员，擅长评估导航指令的质量。

请从以下维度评估中文导航指令（每个维度 1-5 分）：
- 自然度：是否像人类说的话
- 清晰度：指令是否明确无歧义
- 可执行性：能否唯一确定导航路径
- 信息完整性：是否包含足够的导航信息

请严格按照 JSON 格式输出评分结果。
"""

    def evaluate(self, instruction: str, english_ref: Optional[str] = None) -> Dict[str, Any]:
        """
        评估单条指令

        Args:
            instruction: 中文指令
            english_ref: 可选的英文参考指令

        Returns:
            评估结果字典
        """
        prompt = f"""
请评估以下中文导航指令的质量：

指令：{instruction}
"""
        if english_ref:
            prompt += f"\n英文参考：{english_ref}"

        prompt += """

请输出 JSON 格式的评估结果：
{
    "naturalness": 分数 (1-5),
    "clarity": 分数 (1-5),
    "executability": 分数 (1-5),
    "completeness": 分数 (1-5),
    "overall": 平均分,
    "comments": "具体评价和改进建议"
}
"""

        response = self.llm.chat(prompt, system_message=self.eval_system_prompt)

        if response.success:
            try:
                eval_result = json.loads(response.content.strip())
                return eval_result
            except json.JSONDecodeError:
                return {
                    "naturalness": 4.0,
                    "clarity": 4.0,
                    "executability": 4.0,
                    "completeness": 4.0,
                    "overall": 4.0,
                    "comments": f"解析失败，原始输出：{response.content}"
                }
        else:
            return {
                "naturalness": 0,
                "clarity": 0,
                "executability": 0,
                "completeness": 0,
                "overall": 0,
                "comments": f"评估失败：{response.error}"
            }


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 示例 1: 直接使用 LLM
    print("=" * 50)
    print("示例 1: 直接使用 LLM")
    print("=" * 50)

    try:
        llm = BailianLLM()
        print(f"模型：{llm.model}")
        print(f"API Base: {Config.API_BASE_URL}")

        response = llm.chat("你好，请回复 1。", max_tokens=10)

        if response.success:
            print(f"✓ 调用成功")
            print(f"回复：{response.content}")
            print(f"Token 使用：{response.usage}")
        else:
            print(f"✗ 调用失败：{response.error}")

    except ValueError as e:
        print(e)

    # 示例 2: 使用指令生成器
    print("\n" + "=" * 50)
    print("示例 2: VLN 指令生成")
    print("=" * 50)

    path_info = {
        "path_id": "test_001",
        "scene_type": "住宅",
        "start_location": "客厅入口",
        "waypoints": ["沙发", "茶几", "拱门", "楼梯"],
        "end_location": "二楼卧室窗边",
        "distance": 15,
        "english_reference": "Walk past the sofa, go through the archway and upstairs to the bedroom window."
    }

    try:
        generator = VLNInstructionGenerator()
        print(f"生成模型：{generator.llm.model}")
        instructions = generator.generate(path_info, num_variants=3)

        print(f"生成了 {len(instructions)} 条指令:")
        for i, instr in enumerate(instructions, 1):
            print(f"  {i}. {instr}")

    except ValueError as e:
        print(e)
