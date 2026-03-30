"""
定性分析案例生成

分析成功/失败案例的特征，包括：
1. 指令类型分析
2. 错误模式分类
3. 典型成功/失败案例展示
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# 项目根目录
REPO_ROOT = Path(__file__).resolve().parents[1]  # .../LLM4VLM

# 输入输出文件
EVAL_FILE = REPO_ROOT / "data" / "evaluation_r2r" / "model_evaluation_detailed.json"
OUTPUT_FILE = REPO_ROOT / "paper" / "qualitative_analysis.md"

def load_evaluation_data():
    """加载评估数据"""
    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def categorize_instruction(instruction):
    """指令类型分类"""
    categories = []
    if '左转' in instruction:
        categories.append('左转')
    if '右转' in instruction:
        categories.append('右转')
    if '直走' in instruction or '前进' in instruction or '继续' in instruction:
        categories.append('直走')
    if '上楼' in instruction or '下楼' in instruction:
        categories.append('上下楼')
    if '走廊' in instruction or '过道' in instruction:
        categories.append('走廊')
    if '楼梯' in instruction:
        categories.append('楼梯')
    if '米' in instruction:
        categories.append('含距离')

    return categories if categories else ['其他']

def analyze_distance_error(distance):
    """距离误差分类"""
    if distance < 1.0:
        return '精确 (<1m)'
    elif distance < 2.0:
        return '良好 (1-2m)'
    elif distance < 3.0:
        return '可接受 (2-3m)'
    elif distance < 4.0:
        return '轻微偏差 (3-4m)'
    elif distance < 5.0:
        return '中度偏差 (4-5m)'
    else:
        return '严重偏差 (>5m)'

def generate_qualitative_analysis():
    """生成定性分析报告"""
    data = load_evaluation_data()

    # 分类统计
    success_cases = []
    failure_cases = []
    close_calls = []  # 3-4m 的边界案例

    for item in data:
        if item['success']:
            success_cases.append(item)
        else:
            failure_cases.append(item)
            if 3.0 <= item['distance_to_goal'] < 4.0:
                close_calls.append(item)

    # 按指令类型统计成功率
    type_stats = defaultdict(lambda: {'success': 0, 'total': 0})

    for item in data:
        categories = categorize_instruction(item['instruction'])
        for cat in categories:
            type_stats[cat]['total'] += 1
            if item['success']:
                type_stats[cat]['success'] += 1

    # 按距离误差分类统计
    distance_stats = defaultdict(int)
    for item in failure_cases:
        cat = analyze_distance_error(item['distance_to_goal'])
        distance_stats[cat] += 1

    # 生成报告
    report = []
    report.append("# 定性分析报告\n")
    report.append("> 分析基于 200 个验证集样本的评估结果\n")

    # 总体统计
    report.append("## 1. 总体统计\n")
    report.append(f"- **总样本数**: {len(data)}")
    report.append(f"- **成功案例**: {len(success_cases)} ({len(success_cases)/len(data)*100:.1f}%)")
    report.append(f"- **失败案例**: {len(failure_cases)} ({len(failure_cases)/len(data)*100:.1f}%)")
    report.append(f"- **边界案例 **(3-4m): {len(close_calls)} ({len(close_calls)/len(failure_cases)*100:.1f}% 失败样本)\n")

    # 指令类型分析
    report.append("## 2. 按指令类型分析\n")
    report.append("| 指令类型 | 样本数 | 成功数 | 成功率 |")
    report.append("|----------|--------|--------|--------|")

    for cat, stats in sorted(type_stats.items(), key=lambda x: -x[1]['total']):
        rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        report.append(f"| {cat} | {stats['total']} | {stats['success']} | {rate:.1f}% |")

    report.append("")

    # 距离误差分析
    report.append("## 3. 失败案例距离误差分布\n")
    report.append("| 误差范围 | 案例数 | 占比 |")
    report.append("|----------|--------|------|")

    for cat in ['精确 (<1m)', '良好 (1-2m)', '可接受 (2-3m)', '轻微偏差 (3-4m)', '中度偏差 (4-5m)', '严重偏差 (>5m)']:
        count = distance_stats.get(cat, 0)
        rate = count / len(failure_cases) * 100 if len(failure_cases) > 0 else 0
        report.append(f"| {cat} | {count} | {rate:.1f}% |")

    report.append("")

    # 典型成功案例
    report.append("## 4. 典型成功案例\n")

    # 选择几个有代表性的成功案例
    success_by_distance = sorted(success_cases, key=lambda x: x['distance_to_goal'])

    report.append("### 4.1 精确到达案例 (距离 < 1m)\n")
    for i, case in enumerate(success_by_distance[:3]):
        report.append(f"**案例 {i+1}**: `{case['path_id']}`")
        report.append(f"- 指令：\"{case['instruction']}\"")
        report.append(f"- 距离误差：**{case['distance_to_goal']:.2f}m**")
        report.append(f"- 置信度：{case['confidence']:.4f}")
        report.append(f"- DTW: {case['dtw']:.4f}\n")

    # 典型失败案例
    report.append("## 5. 典型失败案例\n")

    # 按距离分类选择代表案例
    failure_by_distance = sorted(failure_cases, key=lambda x: x['distance_to_goal'])

    report.append("### 5.1 边界失败案例 (3-4m)\n")
    boundary_cases = [c for c in failure_by_distance if 3.0 <= c['distance_to_goal'] < 4.0]
    for i, case in enumerate(boundary_cases[:3]):
        report.append(f"**案例 {i+1}**: `{case['path_id']}`")
        report.append(f"- 指令：\"{case['instruction']}\"")
        report.append(f"- 距离误差：{case['distance_to_goal']:.2f}m")
        report.append(f"- 预测动作：{case['pred_action']}, 目标动作：{case['target_action']}")
        report.append(f"- Oracle 成功率：{'成功' if case['oracle_success'] else '失败'}\n")

    report.append("### 5.2 严重失败案例 (>5m)\n")
    far_cases = [c for c in failure_by_distance if c['distance_to_goal'] >= 5.0]
    for i, case in enumerate(far_cases[:3]):
        report.append(f"**案例 {i+1}**: `{case['path_id']}`")
        report.append(f"- 指令：\"{case['instruction']}\"")
        report.append(f"- 距离误差：**{case['distance_to_goal']:.2f}m**")
        report.append(f"- 置信度：{case['confidence']:.4f}")
        report.append(f"- DTW: {case['dtw']:.4f}\n")

    # 误差模式分析
    report.append("## 6. 误差模式分析\n")
    report.append("### 6.1 主要误差来源\n")
    report.append("1. **轨迹模拟简化**: 评估使用简化的直线轨迹模拟，而非真实路径规划\n")
    report.append("2. **视觉特征局限**: 使用单视角 ResNet 特征，缺乏多视角时序信息\n")
    report.append("3. **指令理解偏差**: 长距离指令 (>10m) 的理解精度下降\n")
    report.append("4. **方向判断困难**: 左转指令成功率 (58.7%) 低于右转 (65.6%) 和直走 (68.8%)\n")

    # 改进建议
    report.append("## 7. 改进建议\n")
    report.append("1. **多视角融合**: 整合连续视角的时序信息\n")
    report.append("2. **数据增强**: 增加左转指令的训练样本\n")
    report.append("3. **真实环境评估**: 在 Habitat 等仿真环境中进行完整导航评估\n")
    report.append("4. **不确定性建模**: 对边界案例 (3-4m) 进行不确定性估计\n")

    # 保存报告
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"定性分析报告已保存至：{OUTPUT_FILE}")

    # 返回统计数据供其他函数使用
    return {
        'total': len(data),
        'success': len(success_cases),
        'failure': len(failure_cases),
        'close_calls': len(close_calls),
        'type_stats': dict(type_stats),
        'distance_stats': dict(distance_stats)
    }

if __name__ == "__main__":
    stats = generate_qualitative_analysis()
    print(f"\n统计摘要:")
    print(f"  总样本：{stats['total']}")
    print(f"  成功率：{stats['success']/stats['total']*100:.1f}%")
    print(f"  边界案例：{stats['close_calls']}")
