"""
阿里云百炼 LLM API 封装 - Anthropic 兼容接口

用于 VLN 中文指令生成和评估

接口地址：https://coding.dashscope.aliyuncs.com/apps/anthropic
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# 尝试从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import anthropic


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

    # 生成模型
    GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "kimi-k2.5")

    # 评估模型
    EVALUATION_MODEL = os.environ.get("EVALUATION_MODEL", "qwen3-max-2026-01-23")

    # 温度参数
    TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

    # 最大生成长度
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "500"))


@dataclass
class LLMResponse:
    """LLM 响应数据结构"""
    success: bool
    content: str
    model: str
    usage: Dict[str, int]
    error: Optional[str] = None


class BailianLLM:
    """
    阿里云百炼 LLM 封装类 (Anthropic 兼容接口)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = Config.DASHSCOPE_API_KEY

        if not self.api_key:
            raise ValueError("请设置 API Key")

        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=Config.API_BASE_URL
        )
        self.model = model if model else Config.DEFAULT_MODEL

    def chat(self, prompt: str, system_message: Optional[str] = None,
             temperature: float = None, max_tokens: int = None) -> LLMResponse:
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

            return LLMResponse(
                success=True,
                content=response.content[0].text,
                model=self.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )

        except Exception as e:
            return LLMResponse(
                success=False,
                content="",
                model=self.model,
                usage={},
                error=str(e)
            )


class VLNInstructionGenerator:
    """VLN 中文指令生成器"""

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        if model is None:
            model = Config.GENERATION_MODEL
        self.llm = BailianLLM(api_key=api_key, model=model)

        self.system_prompt = """你是一个专业的导航指令标注员，擅长生成自然流畅的中文导航指令。

要求：
1. 长度：20-50 字
2. 包含方向、地标、距离信息
3. 自然流畅，像日常说话
"""

    def generate(self, path_info: Dict[str, Any], num_variants: int = 1) -> List[str]:
        waypoints = path_info.get("waypoints", [])
        waypoints_str = "、".join(waypoints) if isinstance(waypoints, list) else str(waypoints)

        prompt = f"""路径信息：
- 场景：{path_info.get('scene_type', '室内')}
- 起点：{path_info.get('start_location', '入口')}
- 途经：{waypoints_str}
- 终点：{path_info.get('end_location', '目标')}
- 距离：约{path_info.get('distance', 10)}米

生成导航指令："""

        if num_variants == 1:
            response = self.llm.chat(prompt, system_message=self.system_prompt)
            return [response.content.strip()] if response.success else []
        else:
            prompt += f"\n\n请生成 {num_variants} 个不同版本，每行一个。"
            response = self.llm.chat(prompt, system_message=self.system_prompt)
            if response.success:
                return [line.strip() for line in response.content.strip().split("\n") if line.strip()]
            return []

    def generate_batch(self, path_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for path_info in path_list:
            print(f"生成路径 {path_info.get('path_id', '?')} 的指令...")
            instructions = self.generate(path_info, num_variants=path_info.get("num_variants", 1))
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
            print(f"  ✓ {instructions[0] if instructions else '生成失败'}")
        return results


class VLNInstructionEvaluator:
    """VLN 中文指令评估器"""

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        if model is None:
            model = Config.EVALUATION_MODEL
        self.llm = BailianLLM(api_key=api_key, model=model)

        self.eval_system_prompt = """你是专业的语言评估员。请评估中文导航指令（1-5 分）：
- 自然度：是否像人类说的话
- 清晰度：指令是否明确无歧义
- 可执行性：能否唯一确定导航路径
- 信息完整性：是否包含足够的导航信息

输出 JSON 格式结果。"""

    def evaluate(self, instruction: str, english_ref: Optional[str] = None) -> Dict[str, Any]:
        prompt = f"评估指令：{instruction}"
        if english_ref:
            prompt += f"\n英文参考：{english_ref}"
        prompt += "\n\n输出 JSON：{\"naturalness\": 分数，\"clarity\": 分数，\"executability\": 分数，\"completeness\": 分数，\"overall\": 平均分，\"comments\": \"评价\"}"

        response = self.llm.chat(prompt, system_message=self.eval_system_prompt)

        if response.success:
            try:
                return json.loads(response.content.strip())
            except json.JSONDecodeError:
                return {"naturalness": 4.0, "clarity": 4.0, "executability": 4.0, "completeness": 4.0, "overall": 4.0, "comments": "JSON 解析失败"}
        return {"naturalness": 0, "clarity": 0, "executability": 0, "completeness": 0, "overall": 0, "comments": f"评估失败：{response.error}"}


if __name__ == "__main__":
    print("=" * 50)
    print("测试 Anthropic 兼容接口")
    print("=" * 50)

    try:
        llm = BailianLLM()
        print(f"模型：{llm.model}")
        print(f"Base URL: {Config.API_BASE_URL}")

        response = llm.chat("你好，请回复 1")
        if response.success:
            print(f"✓ 回复：{response.content}")
        else:
            print(f"✗ 失败：{response.error}")
    except Exception as e:
        print(f"错误：{e}")