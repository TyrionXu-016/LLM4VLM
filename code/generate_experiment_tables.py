#!/usr/bin/env python3
"""
生成论文中的对比和消融实验表格
"""

import json
from pathlib import Path
from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../LLM4VLM


def load_experiment_results():
    """加载实验结果"""
    ablation_dir = REPO_ROOT / "experiments" / "ablation_studies"
    comparison_dir = REPO_ROOT / "experiments" / "comparison_studies"

    ablation_results = []
    comparison_results = []

    # 加载消融实验结果
    ablation_summary = ablation_dir / "ablation_summary.json"
    if ablation_summary.exists():
        with open(ablation_summary, 'r', encoding='utf-8') as f:
            data = json.load(f)
            ablation_results = data.get('results', [])

    # 加载对比实验结果
    comparison_summary = comparison_dir / "comparison_summary.json"
    if comparison_summary.exists():
        with open(comparison_summary, 'r', encoding='utf-8') as f:
            data = json.load(f)
            comparison_results = data.get('results', [])

    return ablation_results, comparison_results


def generate_model_ablation_table(results):
    """生成模型架构消融表格"""
    print("\n" + "="*80)
    print("表 1: 模型架构消融实验")
    print("="*80)

    # 基线
    baseline = next((r for r in results if r['name'] == 'baseline'), None)
    baseline_loss = baseline['best_val_loss'] if baseline else 2.63

    table_data = []
    for r in results:
        if r['name'] in ['1_layer', '4_layer', '4_heads', '16_heads', 'd_model_128', 'd_model_512']:
            if 'error' not in r:
                delta = ((r['best_val_loss'] - baseline_loss) / baseline_loss) * 100
                table_data.append([
                    r['name'].replace('_', ' '),
                    f"{r['best_val_loss']:.4f}",
                    f"{r['best_val_acc']:.4f}",
                    f"{r['description']}"
                ])

    print(tabulate(table_data,
                   headers=['模型配置', '验证损失', '验证准确率', '说明'],
                   tablefmt='grid'))


def generate_training_ablation_table(results):
    """生成训练策略消融表格"""
    print("\n" + "="*80)
    print("表 2: 训练策略消融实验")
    print("="*80)

    baseline = next((r for r in results if r['name'] == 'baseline'), None)
    baseline_loss = baseline['best_val_loss'] if baseline else 2.63

    table_data = []
    for r in results:
        if r['name'] in ['lr_5e-5', 'lr_2e-4']:
            if 'error' not in r:
                table_data.append([
                    r['name'].replace('_', ' '),
                    f"{r['best_val_loss']:.4f}",
                    f"{r['best_val_acc']:.4f}",
                    f"{r['description']}"
                ])

    print(tabulate(table_data,
                   headers=['学习率', '验证损失', '验证准确率', '说明'],
                   tablefmt='grid'))


def generate_data_ablation_table(results):
    """生成数据量消融表格"""
    print("\n" + "="*80)
    print("表 3: 数据量消融实验")
    print("="*80)

    table_data = []
    for r in results:
        if r['name'] in ['data_250', 'data_500', 'baseline', 'data_1500']:
            if 'error' not in r:
                samples = r['config'].get('train_samples', 1000) if 'config' in r else 1000
                table_data.append([
                    f"{samples}",
                    f"{r['best_val_loss']:.4f}",
                    f"{r['best_val_acc']:.4f}",
                    f"{r['description']}"
                ])

    print(tabulate(table_data,
                   headers=['训练样本数', '验证损失', '验证准确率', '说明'],
                   tablefmt='grid'))


def generate_feature_ablation_table(results):
    """生成视觉特征消融表格"""
    print("\n" + "="*80)
    print("表 4: 视觉特征消融实验")
    print("="*80)

    baseline = next((r for r in results if r['name'] == 'baseline'), None)
    baseline_loss = baseline['best_val_loss'] if baseline else 2.63

    random_feature = next((r for r in results if r['name'] == 'random_feature'), None)

    table_data = []

    # ResNet 特征（基线）
    if baseline:
        table_data.append([
            "ResNet-50 (预训练)",
            f"{baseline['best_val_loss']:.4f}",
            f"{baseline['best_val_acc']:.4f}",
            "预训练 ImageNet 特征"
        ])

    # 随机特征
    if random_feature and 'error' not in random_feature:
        delta = ((random_feature['best_val_loss'] - baseline_loss) / baseline_loss) * 100
        table_data.append([
            "随机高斯",
            f"{random_feature['best_val_loss']:.4f}",
            f"{random_feature['best_val_acc']:.4f}",
            f"随机初始化特征 (+{delta:.1f}% 损失)"
        ])

    print(tabulate(table_data,
                   headers=['特征类型', '验证损失', '验证准确率', '说明'],
                   tablefmt='grid'))


def generate_instruction_length_table(results):
    """生成指令长度对比表格"""
    print("\n" + "="*80)
    print("表 5: 指令长度对比实验")
    print("="*80)

    table_data = []
    for r in results:
        if r.get('experiment_type') == 'instruction_length' and 'error' not in r:
            table_data.append([
                r['name'].replace('_', ' '),
                f"{r['best_val_loss']:.4f}",
                f"{r['best_val_acc']:.4f}",
                r['description']
            ])

    print(tabulate(table_data,
                   headers=['指令类型', '验证损失', '验证准确率', '说明'],
                   tablefmt='grid'))


def generate_landmark_count_table(results):
    """生成地标数量对比表格"""
    print("\n" + "="*80)
    print("表 6: 地标数量对比实验")
    print("="*80)

    table_data = []
    for r in results:
        if r.get('experiment_type') == 'landmark_count' and 'error' not in r:
            table_data.append([
                r['name'].replace('_', ' '),
                f"{r['best_val_loss']:.4f}",
                f"{r['best_val_acc']:.4f}",
                r['description']
            ])

    print(tabulate(table_data,
                   headers=['地标配置', '验证损失', '验证准确率', '说明'],
                   tablefmt='grid'))


def generate_summary_table(ablation_results, comparison_results):
    """生成汇总表格（用于论文）"""
    print("\n" + "="*80)
    print("表 7: 消融实验汇总（论文格式）")
    print("="*80)

    # 汇总所有实验的验证损失
    all_results = ablation_results + comparison_results

    # 找到最佳结果
    valid_results = [r for r in all_results if 'error' not in r and 'best_val_loss' in r]
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['best_val_loss'])
        print(f"\n最佳验证损失：{best_result['best_val_loss']:.4f} ({best_result['name']})")

    # 论文格式表格数据
    print("\n\\begin{table}[t]")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("\\textbf{Model Configuration} & \\textbf{Val Loss} & \\textbf{Val Acc} \\\\")
    print("\\hline")

    # 添加关键实验结果
    key_experiments = ['baseline', '1_layer', '4_layer', 'random_feature', 'data_250', 'data_500']
    for name in key_experiments:
        result = next((r for r in ablation_results if r['name'] == name), None)
        if result and 'error' not in result:
            print(f"{name.replace('_', ' ').title()} & {result['best_val_loss']:.4f} & {result['best_val_acc']:.4f} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Ablation Study Results}")
    print("\\end{table}")


def main():
    """主函数"""
    print("="*80)
    print("LLM4VLM 对比和消融实验结果汇总")
    print("="*80)

    ablation_results, comparison_results = load_experiment_results()

    if not ablation_results and not comparison_results:
        print("\n未找到实验结果。请先运行消融实验和对比实验脚本。")
        print("运行命令:")
        print("  python code/run_ablation_studies.py")
        print("  python code/run_comparison_experiments.py")
        return

    print(f"\n加载了 {len(ablation_results)} 个消融实验结果")
    print(f"加载了 {len(comparison_results)} 个对比实验结果")

    # 生成各个表格
    generate_model_ablation_table(ablation_results)
    generate_training_ablation_table(ablation_results)
    generate_data_ablation_table(ablation_results)
    generate_feature_ablation_table(ablation_results)
    generate_instruction_length_table(comparison_results)
    generate_landmark_count_table(comparison_results)
    generate_summary_table(ablation_results, comparison_results)

    print("\n" + "="*80)
    print("表格生成完成！")
    print("="*80)


if __name__ == "__main__":
    main()
