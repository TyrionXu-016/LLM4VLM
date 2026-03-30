"""
评估训练好的 VLN 模型（SR/SPL 指标）

使用测试集评估模型的导航性能：
- SR (Success Rate): 成功率
- SPL (Success weighted by Path Length): 效率加权成功率
- DTW: 轨迹相似度
"""

import os
import sys
import json
import torch
from typing import Dict, List, Tuple
from pathlib import Path
from dataclasses import asdict
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../LLM4VLM
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from vln_baseline_model import VLNBaseline, create_model
from vln_evaluation import VLNEvaluator, print_evaluation_report


class VLNModelEvaluator:
    """VLN 模型评估器"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: 模型权重文件路径
            device: 运行设备
        """
        self.device = device
        self.model = None
        self.vocab_size = 0
        self.evaluator = VLNEvaluator(success_distance=3.0)

        # 投影矩阵（2048 -> 256）
        self.proj_matrix = None

        # 加载模型
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载训练好的模型"""
        print(f"加载模型：{model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # 从 checkpoint 推断词表大小
        state_dict = checkpoint['model_state_dict']
        embedding_weight = state_dict.get('instruction_encoder.embedding.weight', None)
        saved_vocab_size = embedding_weight.shape[0] if embedding_weight is not None else 88

        # 加载词表文件获取实际词表大小
        vocab_file = REPO_ROOT / "data" / "r2r_enhanced" / "vocabulary.json"
        if vocab_file.exists():
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.char_to_id = json.load(f)
            self.vocab_size = len(self.char_to_id)
        else:
            self.vocab_size = saved_vocab_size
            self.char_to_id = {'<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3}

        print(f"  保存的词表大小：{saved_vocab_size}")
        print(f"  实际词表大小：{self.vocab_size}")

        # 创建模型（使用实际词表大小）
        self.model = create_model(vocab_size=self.vocab_size, d_model=256)

        # 加载权重（处理嵌入层大小不匹配）
        if self.vocab_size != saved_vocab_size:
            print(f"  ⚠ 词表大小不匹配，跳过嵌入层加载")
            # 保存嵌入层权重
            saved_embedding = state_dict.get('instruction_encoder.embedding.weight')
            del state_dict['instruction_encoder.embedding.weight']
            # 加载其他权重
            self.model.load_state_dict(state_dict, strict=False)
            # 恢复部分嵌入层权重（重叠部分）
            if saved_embedding is not None:
                with torch.no_grad():
                    min_size = min(saved_embedding.shape[0], self.vocab_size)
                    self.model.instruction_encoder.embedding.weight[:min_size] = saved_embedding[:min_size]
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to(self.device)
        self.model.eval()

        # 初始化投影矩阵（与训练时一致）
        torch.manual_seed(42)
        self.proj_matrix = torch.randn(2048, 256) * 0.02

        print(f"  ✓ 模型加载成功")

    def predict_action(self, batch: Dict[str, torch.Tensor]) -> Tuple[int, float]:
        """
        预测动作

        Returns:
            (action_id, confidence)
        """
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.no_grad():
            # 使用 model.predict 方法
            action, confidence = self.model.predict(batch)

        return action.item(), confidence.item()

    def _simulate_trajectory(self, start_path: List[List[float]],
                             target_action: int,
                             num_views: int = 36) -> List[List[float]]:
        """
        根据预测动作模拟轨迹

        Args:
            start_path: 起始路径点
            target_action: 预测的目标动作（0-35）
            num_views: 视角数

        Returns:
            模拟轨迹
        """
        # 简化的轨迹模拟：根据动作选择对应的视角方向
        # 实际应用中需要在仿真环境中执行动作

        if len(start_path) == 0:
            return [[0, 0, 0]]

        # 将动作映射到视角索引
        # 动作 0-3: 前方，8-12: 左侧，24-28: 右侧，16-20: 后方
        view_offset = target_action % num_views

        # 模拟轨迹：从起始点沿预测方向移动
        trajectory = [start_path[0].copy()]

        # 根据动作类型添加轨迹点
        if 0 <= target_action <= 4:  # 前方
            for i in range(1, min(4, len(start_path))):
                trajectory.append(start_path[i].copy())
        elif 8 <= target_action <= 12:  # 左侧
            # 模拟向左转并前进
            for i in range(1, min(3, len(start_path))):
                pos = start_path[i].copy()
                pos[0] -= 0.5 * i  # 向左偏移
                trajectory.append(pos)
        elif 24 <= target_action <= 28:  # 右侧
            # 模拟向右转并前进
            for i in range(1, min(3, len(start_path))):
                pos = start_path[i].copy()
                pos[0] += 0.5 * i  # 向右偏移
                trajectory.append(pos)
        else:  # 其他方向：沿原路径
            for i in range(1, min(3, len(start_path))):
                trajectory.append(start_path[i].copy())

        return trajectory

    def evaluate_single(self, sample: Dict) -> Dict:
        """
        评估单个样本

        Args:
            sample: 测试样本（包含 instruction, visual_features, candidate_directions, path 等）

        Returns:
            评估结果字典
        """
        # 直接使用数据中的 instruction_ids（与训练时一致）
        # 注意：数据中的 ID 已经是基于训练词表的，无需重新映射
        instr_ids = sample['instruction_ids']

        # 准备输入
        instr_ids_tensor = torch.tensor([2] + instr_ids + [3],  # [CLS] + ids + [SEP]
                                         dtype=torch.long).unsqueeze(0)
        visual_feat = torch.tensor(sample['visual_features'],
                                    dtype=torch.float32).unsqueeze(0)
        visual_feat = visual_feat.view(1, -1, 2048)

        # 投影候选方向 (2048 -> 256)
        candidate_dirs_raw = torch.tensor(sample['candidate_directions'],
                                           dtype=torch.float32)
        candidate_dirs = torch.matmul(candidate_dirs_raw, self.proj_matrix).unsqueeze(0)

        instr_mask = torch.zeros(1, instr_ids_tensor.shape[1], dtype=torch.bool)

        batch = {
            'instructions': instr_ids_tensor,
            'visual_features': visual_feat,
            'candidate_directions': candidate_dirs,
            'instruction_mask': instr_mask
        }

        # 预测动作
        pred_action, confidence = self.predict_action(batch)

        # 获取真实信息
        ref_path = sample['path']  # 参考路径
        target_action = sample['target_action']  # 真实目标动作
        path_length = sample.get('path_length', 10.0)

        # 模拟轨迹（根据预测动作）
        trajectory = self._simulate_trajectory(ref_path, pred_action)

        # 计算目标位置（参考路径终点）
        goal_position = ref_path[-1] if ref_path else [0, 0, 0]

        # 评估
        result = self.evaluator.evaluate_single(
            trajectory=trajectory,
            reference_path=ref_path,
            goal_position=goal_position,
            path_id=sample.get('path_id', 'unknown'),
            instruction=sample.get('instruction', '')
        )

        return {
            'path_id': result.path_id,
            'instruction': result.instruction,
            'pred_action': pred_action,
            'target_action': target_action,
            'confidence': confidence,
            'success': result.success,
            'distance_to_goal': result.distance_to_goal,
            'spl': result.spl,
            'dtw': result.dtw_distance,
            'oracle_success': result.oracle_success
        }

    def evaluate_dataset(self, data_file: str,
                         max_samples: int = None) -> Tuple[List[Dict], Dict]:
        """
        评估整个数据集

        Args:
            data_file: 测试数据文件
            max_samples: 最大评估样本数（None 表示全部）

        Returns:
            (详细结果列表，聚合指标字典)
        """
        print(f"\n评估数据集：{data_file}")

        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        print(f"  样本数：{len(data)}")

        # 逐个评估
        results = []
        for i, sample in enumerate(data):
            if (i + 1) % 50 == 0:
                print(f"  进度：{i+1}/{len(data)}")

            result = self.evaluate_single(sample)
            results.append(result)

        # 聚合指标
        metrics = self._aggregate_results(results)

        return results, metrics

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """聚合评估结果"""
        n = len(results)

        if n == 0:
            return {}

        # 准确率（动作预测）
        correct_actions = sum(1 for r in results
                              if r['pred_action'] == r['target_action'])
        action_accuracy = correct_actions / n

        # SR
        num_success = sum(1 for r in results if r['success'])
        sr = num_success / n

        # SPL
        avg_spl = sum(r['spl'] for r in results) / n

        # Oracle SR
        num_oracle = sum(1 for r in results if r['oracle_success'])
        oracle_sr = num_oracle / n

        # 平均置信度
        avg_confidence = sum(r['confidence'] for r in results) / n

        # 平均距离误差
        avg_distance = sum(r['distance_to_goal'] for r in results) / n

        # DTW
        avg_dtw = sum(r['dtw'] for r in results) / n

        return {
            'num_samples': n,
            'action_accuracy': action_accuracy,
            'SR': sr,
            'SPL': avg_spl,
            'Oracle_SR': oracle_sr,
            'avg_confidence': avg_confidence,
            'avg_distance_error': avg_distance,
            'avg_DTW': avg_dtw,
        }


def print_model_evaluation_report(metrics: Dict,
                                   detailed_results: List[Dict]):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("VLN 模型评估报告")
    print("=" * 60)

    print(f"\n评估样本数：{metrics.get('num_samples', 0)}")

    print("\n=== 核心指标 ===")
    print(f"  SR (成功率):          {metrics.get('SR', 0)*100:.2f}%")
    print(f"  SPL (效率加权):       {metrics.get('SPL', 0)*100:.2f}%")
    print(f"  Oracle SR:            {metrics.get('Oracle_SR', 0)*100:.2f}%")

    print("\n=== 动作预测 ===")
    print(f"  动作准确率：{metrics.get('action_accuracy', 0)*100:.2f}%")
    print(f"  平均置信度：{metrics.get('avg_confidence', 0)*100:.2f}%")

    print("\n=== 路径质量 ===")
    print(f"  平均距离误差：{metrics.get('avg_distance_error', 0):.4f} 米")
    print(f"  平均 DTW:     {metrics.get('avg_DTW', 0):.4f}")

    # 按指令类型分析
    print("\n=== 按指令关键词分析 ===")

    # 左转指令
    left_turn = [r for r in detailed_results if '左转' in r['instruction']]
    if left_turn:
        sr_left = sum(1 for r in left_turn if r['success']) / len(left_turn)
        print(f"  左转指令 ({len(left_turn)}条): SR = {sr_left*100:.1f}%")

    # 右转指令
    right_turn = [r for r in detailed_results if '右转' in r['instruction']]
    if right_turn:
        sr_right = sum(1 for r in right_turn if r['success']) / len(right_turn)
        print(f"  右转指令 ({len(right_turn)}条): SR = {sr_right*100:.1f}%")

    # 直走指令
    go_straight = [r for r in detailed_results if '直走' in r['instruction']]
    if go_straight:
        sr_straight = sum(1 for r in go_straight if r['success']) / len(go_straight)
        print(f"  直走指令 ({len(go_straight)}条): SR = {sr_straight*100:.1f}%")

    # 详细结果（前 10 个）
    print("\n" + "=" * 60)
    print("详细结果（前 10 个样本）")
    print("=" * 60)

    for i, result in enumerate(detailed_results[:10]):
        status = "✓" if result['success'] else "✗"
        match = "✓" if result['pred_action'] == result['target_action'] else "✗"
        print(f"\n{i+1}. {result['path_id']} {status}")
        print(f"   指令：{result['instruction'][:40]}...")
        print(f"   预测动作：{result['pred_action']} (目标：{result['target_action']}) {match}")
        print(f"   置信度：{result['confidence']:.4f}")
        print(f"   距离目标：{result['distance_to_goal']:.4f} 米")
        print(f"   SPL: {result['spl']:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(REPO_ROOT / "checkpoints" / "vln_r2r_best.pt"))
    parser.add_argument("--val-data", type=str, default=str(REPO_ROOT / "data" / "r2r_enhanced" / "r2r_enhanced_val.json"))
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "data" / "evaluation_r2r"))
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("VLN 模型评估（SR/SPL）")
    print("=" * 60)

    model_path = args.model_path
    val_data_file = args.val_data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建评估器
    evaluator = VLNModelEvaluator(model_path=model_path, device=args.device)

    # 评估验证集
    results, metrics = evaluator.evaluate_dataset(val_data_file, max_samples=args.max_samples)

    # 打印报告
    print_model_evaluation_report(metrics, results)

    # 保存结果
    metrics_file = output_dir / "model_evaluation_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n指标已保存至：{metrics_file}")

    detailed_file = output_dir / "model_evaluation_detailed.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"详细结果已保存至：{detailed_file}")

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
