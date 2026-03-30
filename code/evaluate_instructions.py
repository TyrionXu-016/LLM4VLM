"""
VLN 中文指令质量评估工具

提供多维度自动评估功能
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class EvaluationResult:
    """评估结果数据结构"""
    path_id: str
    instruction: str

    # 形式指标
    word_count: int
    verb_count: int
    landmark_count: int
    direction_count: int

    # 评分
    naturalness: float  # 自然度
    clarity: float      # 清晰度
    executability: float  # 可执行性
    completeness: float  # 完整性
    overall: float      # 综合分

    # 等级
    quality_level: str
    issues: List[str]


class SimpleChineseVLNEvaluator:
    """
    中文 VLN 指令评估器

    不依赖外部 API，使用规则进行快速评估
    """

    def __init__(self):
        # 方向词库
        self.direction_words = [
            "左", "右", "前", "后", "直", "东", "南", "西", "北",
            "进来", "进去", "出去", "上来", "下去", "过来", "过去"
        ]

        # 动作词库
        self.verb_words = [
            "走", "跑", "转", "拐", "穿", "过", "进", "出",
            "上", "下", "停", "到", "经过", "路过", "穿过", "走过"
        ]

        # 常见地标类型词
        self.landmark_words = [
            "门", "窗", "桌", "椅", "沙发", "床", "楼梯", "电梯",
            "厨房", "卫生间", "卧室", "客厅", "餐厅", "阳台",
            "画", "灯", "柜子", "冰箱", "电视", "镜子",
            "走廊", "过道", "大厅", "房间", "尽头", "入口", "出口"
        ]

    def evaluate_single(self, instruction: str, path_id: str = "") -> EvaluationResult:
        """评估单条指令"""

        # 1. 形式指标
        word_count = len(instruction)

        verb_count = sum(1 for v in self.verb_words if v in instruction)
        landmark_count = sum(1 for l in self.landmark_words if l in instruction)
        direction_count = sum(1 for d in self.direction_words if d in instruction)

        # 2. 规则评分
        naturalness = self._rate_naturalness(instruction)
        clarity = self._rate_clarity(instruction)
        executability = self._rate_executability(instruction)
        completeness = self._rate_completeness(instruction, verb_count, landmark_count, direction_count)

        # 3. 综合分
        overall = (naturalness + clarity + executability + completeness) / 4

        # 4. 质量等级
        quality_level = self._score_to_level(overall)

        # 5. 问题检测
        issues = self._detect_issues(instruction, word_count, verb_count, landmark_count, direction_count)

        return EvaluationResult(
            path_id=path_id,
            instruction=instruction,
            word_count=word_count,
            verb_count=verb_count,
            landmark_count=landmark_count,
            direction_count=direction_count,
            naturalness=naturalness,
            clarity=clarity,
            executability=executability,
            completeness=completeness,
            overall=overall,
            quality_level=quality_level,
            issues=issues
        )

    def _rate_naturalness(self, instruction: str) -> float:
        """
        自然度评分 (1-5 分)

        基于：
        - 是否有机器翻译痕迹
        - 是否使用口语化表达
        """
        score = 3.0  # 基础分

        # 加分项
        if "后" in instruction and ("走" in instruction or "转" in instruction):
            score += 0.5  # "看到 X 后左转" 是自然的中文表达

        if "就" in instruction:
            score += 0.5  # "就到了" 口语化

        if "路过" in instruction or "穿过" in instruction:
            score += 0.3  # 地道的动词

        # 减分项
        if instruction.count("然后") > 2:
            score -= 0.5  # 过多"然后"显得机械

        if instruction.startswith("请"):
            score -= 0.3  # "请"字开头像机器指令

        return min(5.0, max(1.0, score))

    def _rate_clarity(self, instruction: str) -> float:
        """
        清晰度评分 (1-5 分)

        基于：
        - 方向词是否明确
        - 是否有歧义
        """
        score = 3.0

        # 有明确方向词
        if any(d in instruction for d in ["左转", "右转", "直走", "向左", "向右"]):
            score += 1.0

        # 有地标参考
        landmark_refs = ["沙发", "桌子", "门", "窗", "楼梯", "画", "柜子"]
        if any(l in instruction for l in landmark_refs):
            score += 0.5

        # 有终点描述
        if "停下" in instruction or "到" in instruction or "尽头" in instruction:
            score += 0.5

        return min(5.0, max(1.0, score))

    def _rate_executability(self, instruction: str) -> float:
        """
        可执行性评分 (1-5 分)

        基于：
        - 指令是否可唯一确定路径
        - 是否有足够的导航信息
        """
        score = 3.0

        # 有起点暗示
        if "从" in instruction or "进" in instruction:
            score += 0.5

        # 有连续动作
        if instruction.count("，") >= 2:
            score += 0.5  # 多步指令

        # 有地标序列
        landmark_count = sum(1 for l in self.landmark_words if l in instruction)
        if landmark_count >= 3:
            score += 1.0
        elif landmark_count >= 1:
            score += 0.5

        return min(5.0, max(1.0, score))

    def _rate_completeness(self, instruction: str, verb_count: int, landmark_count: int, direction_count: int) -> float:
        """
        完整性评分 (1-5 分)

        基于：
        - 字数
        - 动词、地标、方向词数量
        """
        score = 3.0

        # 字数检查
        if 25 <= len(instruction) <= 50:
            score += 1.0
        elif 20 <= len(instruction) < 25 or 50 < len(instruction) <= 60:
            score += 0.5

        # 动词数量
        if verb_count >= 3:
            score += 0.5
        elif verb_count >= 1:
            score += 0.3

        # 地标数量
        if landmark_count >= 3:
            score += 0.5
        elif landmark_count >= 1:
            score += 0.3

        # 方向词数量
        if direction_count >= 3:
            score += 0.5
        elif direction_count >= 1:
            score += 0.3

        return min(5.0, max(1.0, score))

    def _score_to_level(self, score: float) -> str:
        """分数转等级"""
        if score >= 4.5:
            return "优秀"
        elif score >= 3.5:
            return "良好"
        elif score >= 2.5:
            return "合格"
        else:
            return "不合格"

    def _detect_issues(self, instruction: str, word_count: int, verb_count: int,
                       landmark_count: int, direction_count: int) -> List[str]:
        """检测潜在问题"""
        issues = []

        if word_count < 20:
            issues.append("字数过短（<20 字）")
        elif word_count > 60:
            issues.append("字数过长（>60 字）")

        if verb_count == 0:
            issues.append("缺少动作词")

        if landmark_count == 0:
            issues.append("缺少地标参考")

        if direction_count == 0:
            issues.append("缺少方向信息")

        if instruction.count("然后") > 3:
            issues.append("过多使用'然后'")

        return issues

    def batch_evaluate(self, instructions: List[Dict]) -> List[EvaluationResult]:
        """批量评估"""
        results = []
        for item in instructions:
            result = self.evaluate_single(
                item.get("instruction", ""),
                item.get("path_id", "")
            )
            results.append(result)
        return results

    def print_report(self, results: List[EvaluationResult]):
        """打印评估报告"""
        print("=" * 70)
        print("VLN 中文指令质量评估报告")
        print("=" * 70)
        print()

        # 总体统计
        total = len(results)
        excellent = sum(1 for r in results if r.quality_level == "优秀")
        good = sum(1 for r in results if r.quality_level == "良好")
        passable = sum(1 for r in results if r.quality_level == "合格")
        failed = sum(1 for r in results if r.quality_level == "不合格")

        avg_overall = sum(r.overall for r in results) / total if total > 0 else 0

        print(f"指令总数：{total}")
        print(f"平均综合分：{avg_overall:.2f} / 5.0")
        print()
        print("质量分布:")
        print(f"  优秀：{excellent} ({excellent/total*100:.1f}%)")
        print(f"  良好：{good} ({good/total*100:.1f}%)")
        print(f"  合格：{passable} ({passable/total*100:.1f}%)")
        print(f"  不合格：{failed} ({failed/total*100:.1f}%)")
        print()

        # 逐条详情
        print("-" * 70)
        print("逐条评估详情:")
        print("-" * 70)

        for r in results:
            print(f"\n【路径 {r.path_id}】等级：{r.quality_level}")
            print(f"指令：{r.instruction}")
            print(f"字数：{r.word_count} | 动词：{r.verb_count} | 地标：{r.landmark_count} | 方向：{r.direction_count}")
            print(f"自然度：{r.naturalness:.1f} | 清晰度：{r.clarity:.1f} | 可执行：{r.executability:.1f} | 完整：{r.completeness:.1f}")
            print(f"综合分：{r.overall:.2f}")

            if r.issues:
                print(f"问题：{', '.join(r.issues)}")


# === 使用示例 ===
if __name__ == "__main__":
    # 加载新生成的指令
    with open("/Users/tyrion/Projects/Papers/data/generated_instructions.json", "r", encoding="utf-8") as f:
        instructions = json.load(f)

    # 创建评估器
    evaluator = SimpleChineseVLNEvaluator()

    # 批量评估
    results = evaluator.batch_evaluate(instructions)

    # 打印报告
    evaluator.print_report(results)

    # 保存详细结果
    output_data = [asdict(r) for r in results]
    output_file = "/Users/tyrion/Projects/Papers/data/evaluation_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到：{output_file}")
