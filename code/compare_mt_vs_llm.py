"""
对比分析：机器翻译 vs LLM 直接生成

评估两种方法生成的中文指令质量差异
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../LLM4VLM
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evaluate_instructions import SimpleChineseVLNEvaluator


def compare_translations(machine_trans_file: str, llm_generated_file: str, output_file: str):
    """
    对比机器翻译和 LLM 生成的质量

    Args:
        machine_trans_file: 机器翻译结果文件
        llm_generated_file: LLM 直接生成结果文件
        output_file: 对比分析输出文件
    """
    # 加载数据
    with open(machine_trans_file, 'r', encoding='utf-8') as f:
        mt_data = json.load(f)

    with open(llm_generated_file, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)

    print(f"机器翻译数据：{len(mt_data)} 条")
    print(f"LLM 生成数据：{len(llm_data)} 条")
    print()

    # 创建评估器
    evaluator = SimpleChineseVLNEvaluator()

    # 评估机器翻译
    print("评估机器翻译...")
    mt_results = []
    for item in mt_data:
        result = evaluator.evaluate_single(item['machine_translation'], item['path_id'])
        result.method = "machine_translation"
        result.english_reference = item['english_reference']
        mt_results.append(result)

    # 评估 LLM 生成
    print("评估 LLM 生成...")
    llm_results = evaluator.batch_evaluate(llm_data)

    # 统计分析
    mt_avg = sum(r.overall for r in mt_results) / len(mt_results) if mt_results else 0
    llm_avg = sum(r.overall for r in llm_results) / len(llm_results) if llm_results else 0

    mt_excellent = sum(1 for r in mt_results if r.quality_level == "优秀") / len(mt_results) * 100 if mt_results else 0
    llm_excellent = sum(1 for r in llm_results if r.quality_level == "优秀") / len(llm_results) * 100 if llm_results else 0

    print()
    print("=" * 70)
    print("对比分析结果")
    print("=" * 70)
    print()
    print(f"{'指标':<20} {'机器翻译':<20} {'LLM 生成':<20}")
    print("-" * 60)
    print(f"{'平均综合分':<20} {mt_avg:<20.2f} {llm_avg:<20.2f}")
    print(f"{'优秀率':<20} {mt_excellent:<20.1f}% {llm_excellent:<20.1f}%")
    print()

    # 详细对比
    print("=" * 70)
    print("详细对比")
    print("=" * 70)

    for i, (mt, llm) in enumerate(zip(mt_results, llm_results[:len(mt_results)]), 1):
        print(f"\n【样本 {i}】")
        print(f"英文：{mt.english_reference}")
        print()
        print(f"机器翻译：{mt.instruction}")
        print(f"  分数：{mt.overall:.2f} ({mt.quality_level})")
        print(f"  自然度：{mt.naturalness:.1f} | 清晰度：{mt.clarity:.1f} | 可执行：{mt.executability:.1f}")
        print()
        print(f"LLM 生成：{llm.instruction}")
        print(f"  分数：{llm.overall:.2f} ({llm.quality_level})")
        print(f"  自然度：{llm.naturalness:.1f} | 清晰度：{llm.clarity:.1f} | 可执行：{llm.executability:.1f}")

        # 判断哪个更好
        if mt.overall > llm.overall:
            print(f"  → 机器翻译更好 (+{mt.overall - llm.overall:.2f})")
        elif llm.overall > mt.overall:
            print(f"  → LLM 生成更好 (+{llm.overall - mt.overall:.2f})")
        else:
            print(f"  → 平局")

    # 保存结果
    output_data = {
        "summary": {
            "machine_translation": {
                "avg_overall": mt_avg,
                "excellent_rate": mt_excellent,
                "count": len(mt_results)
            },
            "llm_generated": {
                "avg_overall": llm_avg,
                "excellent_rate": llm_excellent,
                "count": len(llm_results)
            }
        },
        "detailed_comparison": [
            {
                "path_id": mt.path_id,
                "english_reference": mt.english_reference,
                "machine_translation": {
                    "instruction": mt.instruction,
                    "overall": mt.overall,
                    "quality_level": mt.quality_level,
                    "naturalness": mt.naturalness,
                    "clarity": mt.clarity,
                    "executability": mt.executability,
                    "completeness": mt.completeness
                },
                "llm_generated": {
                    "instruction": llm.instruction,
                    "overall": llm.overall,
                    "quality_level": llm.quality_level,
                    "naturalness": llm.naturalness,
                    "clarity": llm.clarity,
                    "executability": llm.executability,
                    "completeness": llm.completeness
                }
            }
            for mt, llm in zip(mt_results, llm_results[:len(mt_results)])
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print()
    print(f"对比结果已保存到：{output_file}")

    return output_data


if __name__ == "__main__":
    compare_translations(
        machine_trans_file=str(REPO_ROOT / "data" / "r2r_machine_translations.json"),
        llm_generated_file=str(REPO_ROOT / "data" / "generated_instructions.json"),
        output_file=str(REPO_ROOT / "data" / "mt_vs_llm_comparison.json")
    )
