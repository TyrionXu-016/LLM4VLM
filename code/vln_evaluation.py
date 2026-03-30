"""
VLN 评估指标实现

实现标准的 VLN 评估指标：
- SR (Success Rate): 成功率
- SPL (Success weighted by Path Length): 成功率 × 路径长度
- SDTW (Soft Dynamic Time Warping): 软动态时间弯曲
- Oracle: 上限估计
"""

import math
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VLNEvaluationResult:
    """VLN 评估结果"""
    path_id: str
    instruction: str
    success: bool
    trajectory_length: float  # 实际行走路径长度
    path_length: float  # 参考路径长度
    distance_to_goal: float  # 到达位置与目标的距离
    dtw_distance: float  # DTW 距离
    oracle_success: bool  # Oracle 成功率
    spl: float  # SPL 值


class VLNEvaluator:
    """VLN 评估器"""

    def __init__(self, success_distance: float = 3.0):
        """
        Args:
            success_distance: 成功判断距离阈值（米），R2R 标准为 3 米
        """
        self.success_distance = success_distance

    def euclidean_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """计算欧几里得距离"""
        if len(pos1) != len(pos2):
            raise ValueError("位置坐标维度不一致")

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

    def trajectory_length(self, trajectory: List[List[float]]) -> float:
        """计算轨迹总长度"""
        if len(trajectory) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(trajectory)):
            total += self.euclidean_distance(trajectory[i-1], trajectory[i])

        return total

    def check_success(self, final_position: List[float],
                      goal_position: List[float]) -> bool:
        """
        检查是否成功到达目标

        Args:
            final_position: 最终位置 [x, y, z]
            goal_position: 目标位置 [x, y, z]

        Returns:
            是否成功
        """
        distance = self.euclidean_distance(final_position, goal_position)
        return distance <= self.success_distance

    def compute_spl(self, success: bool, trajectory_length: float,
                    path_length: float) -> float:
        """
        计算 SPL (Success weighted by Path Length)

        SPL = success × min(trajectory_length, path_length) / trajectory_length

        Args:
            success: 是否成功
            trajectory_length: 实际行走路径长度
            path_length: 参考路径长度（最短路径）

        Returns:
            SPL 值 (0-1)
        """
        if not success or trajectory_length <= 0:
            return 0.0

        # 如果实际路径比参考路径短，使用参考路径长度
        # 这鼓励 Agent 走最短路径
        efficiency = min(path_length, trajectory_length) / trajectory_length

        return efficiency

    def compute_dtw(self, trajectory: List[List[float]],
                    reference_path: List[List[float]]) -> float:
        """
        计算动态时间弯曲距离 (Dynamic Time Warping)

        衡量轨迹与参考路径的相似度

        Args:
            trajectory: 实际行走轨迹
            reference_path: 参考路径

        Returns:
            DTW 距离（越小越好）
        """
        n = len(trajectory)
        m = len(reference_path)

        if n == 0 or m == 0:
            return float('inf')

        # DTW 距离矩阵
        dtw_matrix = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dtw_matrix[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self.euclidean_distance(trajectory[i-1], reference_path[j-1])
                dtw_matrix[i][j] = cost + min(
                    dtw_matrix[i-1][j],     # 插入
                    dtw_matrix[i][j-1],     # 删除
                    dtw_matrix[i-1][j-1]    # 匹配
                )

        # 归一化 DTW 距离
        dtw_distance = dtw_matrix[n][m] / (n + m)

        return dtw_distance

    def compute_normalized_dtw(self, trajectory: List[List[float]],
                                reference_path: List[List[float]]) -> float:
        """
        计算归一化 DTW (0-1，1 表示完全匹配)

        SDTW = 1 - dtw / max_len
        """
        dtw = self.compute_dtw(trajectory, reference_path)
        max_len = max(len(trajectory), len(reference_path))

        if max_len == 0:
            return 0.0

        # 使用指数衰减归一化
        normalized = 1.0 / (1.0 + dtw / max_len)

        return normalized

    def check_oracle_success(self, trajectory: List[List[float]],
                             goal_position: List[float]) -> bool:
        """
        检查 Oracle 成功率

        Oracle：如果轨迹中任何位置到达目标范围内，则计为成功

        Args:
            trajectory: 完整轨迹
            goal_position: 目标位置

        Returns:
            Oracle 成功
        """
        for position in trajectory:
            if self.euclidean_distance(position, goal_position) <= self.success_distance:
                return True
        return False

    def evaluate_single(self,
                        trajectory: List[List[float]],
                        reference_path: List[List[float]],
                        goal_position: List[float],
                        path_id: str = "",
                        instruction: str = "") -> VLNEvaluationResult:
        """
        评估单个样本

        Args:
            trajectory: Agent 实际行走轨迹
            reference_path: 参考路径（R2R 提供的路径）
            goal_position: 目标位置
            path_id: 路径 ID
            instruction: 导航指令

        Returns:
            VLNEvaluationResult
        """
        # 计算各项指标
        traj_length = self.trajectory_length(trajectory)
        path_length = self.trajectory_length(reference_path)
        final_position = trajectory[-1] if trajectory else [0, 0, 0]
        distance_to_goal = self.euclidean_distance(final_position, goal_position)

        # 成功判断
        success = self.check_success(final_position, goal_position)

        # SPL
        spl = self.compute_spl(success, traj_length, path_length)

        # DTW
        dtw = self.compute_dtw(trajectory, reference_path)

        # Oracle
        oracle = self.check_oracle_success(trajectory, goal_position)

        return VLNEvaluationResult(
            path_id=path_id,
            instruction=instruction,
            success=success,
            trajectory_length=traj_length,
            path_length=path_length,
            distance_to_goal=distance_to_goal,
            dtw_distance=dtw,
            oracle_success=oracle,
            spl=spl
        )

    def evaluate_batch(self,
                       trajectories: List[List[List[float]]],
                       reference_paths: List[List[List[float]]],
                       goal_positions: List[List[float]],
                       path_ids: Optional[List[str]] = None,
                       instructions: Optional[List[str]] = None) -> List[VLNEvaluationResult]:
        """
        批量评估

        Args:
            trajectories: 轨迹列表
            reference_paths: 参考路径列表
            goal_positions: 目标位置列表
            path_ids: 路径 ID 列表
            instructions: 指令列表

        Returns:
            评估结果列表
        """
        results = []

        for i in range(len(trajectories)):
            result = self.evaluate_single(
                trajectory=trajectories[i],
                reference_path=reference_paths[i],
                goal_position=goal_positions[i],
                path_id=path_ids[i] if path_ids else f"sample_{i}",
                instruction=instructions[i] if instructions else ""
            )
            results.append(result)

        return results

    def aggregate_metrics(self,
                          results: List[VLNEvaluationResult]) -> Dict[str, float]:
        """
        聚合评估指标

        Args:
            评估结果列表

        Returns:
            聚合指标字典
        """
        if not results:
            return {}

        n = len(results)

        # 成功率
        num_success = sum(1 for r in results if r.success)
        sr = num_success / n

        # 平均 SPL
        avg_spl = sum(r.spl for r in results) / n

        # Oracle 成功率
        num_oracle = sum(1 for r in results if r.oracle_success)
        oracle_sr = num_oracle / n

        # 平均 DTW
        avg_dtw = sum(r.dtw_distance for r in results) / n

        # 归一化 DTW
        avg_normalized_dtw = sum(
            1.0 / (1.0 + r.dtw_distance / max(len([1]), len([1])))
            for r in results
        ) / n

        # 平均距离误差
        avg_distance_error = sum(r.distance_to_goal for r in results) / n

        # 轨迹长度统计
        avg_trajectory_length = sum(r.trajectory_length for r in results) / n
        avg_path_length = sum(r.path_length for r in results) / n

        return {
            'SR': sr,
            'SPL': avg_spl,
            'Oracle_SR': oracle_sr,
            'DTW': avg_dtw,
            'Normalized_DTW': avg_normalized_dtw,
            'Distance_Error': avg_distance_error,
            'Avg_Trajectory_Length': avg_trajectory_length,
            'Avg_Path_Length': avg_path_length,
            'Num_Success': num_success,
            'Num_Total': n,
        }


def print_evaluation_report(metrics: Dict[str, float],
                            detailed_results: Optional[List[VLNEvaluationResult]] = None):
    """
    打印评估报告

    Args:
        metrics: 聚合指标
        detailed_results: 详细结果（可选）
    """
    print("=" * 60)
    print("VLN 评估报告")
    print("=" * 60)

    print(f"\n样本总数：{metrics.get('Num_Total', 0)}")
    print(f"成功数量：{metrics.get('Num_Success', 0)}")

    print("\n主要指标:")
    print(f"  SR (成功率):          {metrics.get('SR', 0)*100:.2f}%")
    print(f"  SPL (效率加权):       {metrics.get('SPL', 0)*100:.2f}%")
    print(f"  Oracle SR:            {metrics.get('Oracle_SR', 0)*100:.2f}%")

    print("\n路径质量:")
    print(f"  DTW 距离：{metrics.get('DTW', 0):.4f}")
    print(f"  归一化 DTW: {metrics.get('Normalized_DTW', 0):.4f}")
    print(f"  距离误差：{metrics.get('Distance_Error', 0):.4f} 米")

    print("\n轨迹统计:")
    print(f"  平均轨迹长度：{metrics.get('Avg_Trajectory_Length', 0):.2f} 米")
    print(f"  平均参考路径：{metrics.get('Avg_Path_Length', 0):.2f} 米")

    if detailed_results:
        print("\n" + "=" * 60)
        print("详细结果（前 10 个样本）")
        print("=" * 60)

        for i, result in enumerate(detailed_results[:10]):
            status = "✓" if result.success else "✗"
            print(f"\n{i+1}. {result.path_id} {status}")
            print(f"   指令：{result.instruction[:50]}...")
            print(f"   距离目标：{result.distance_to_goal:.2f} 米")
            print(f"   SPL: {result.spl:.4f}")
            print(f"   DTW: {result.dtw_distance:.4f}")


def simulate_navigation_predictions(
        model_output_file: Optional[str] = None,
        num_samples: int = 100) -> Tuple[List[VLNEvaluationResult], Dict[str, float]]:
    """
    模拟导航预测并评估（用于测试）

    Args:
        model_output_file: 模型输出文件（可选）
        num_samples: 模拟样本数

    Returns:
        评估结果和聚合指标
    """
    import random

    evaluator = VLNEvaluator(success_distance=3.0)

    results = []

    for i in range(num_samples):
        # 模拟参考路径（7 个视角，总长约 10 米）
        ref_path = [[j, 0, 0] for j in range(7)]

        # 模拟目标位置（路径终点）
        goal = ref_path[-1]

        # 模拟 Agent 轨迹（带有随机偏差）
        trajectory = []
        for j, point in enumerate(ref_path):
            # 添加随机偏差
            noise_x = random.gauss(0, 0.5)
            noise_z = random.gauss(0, 0.5)
            trajectory.append([point[0] + noise_x, point[1], point[2] + noise_z])

        # 模拟成功率（约 50%）
        if random.random() < 0.5:
            # 成功：最终位置接近目标
            trajectory[-1] = [goal[0] + random.gauss(0, 1),
                              goal[1],
                              goal[2] + random.gauss(0, 1)]
        else:
            # 失败：最终位置远离目标
            trajectory[-1] = [goal[0] + random.gauss(5, 2),
                              goal[1],
                              goal[2] + random.gauss(5, 2)]

        result = evaluator.evaluate_single(
            trajectory=trajectory,
            reference_path=ref_path,
            goal_position=goal,
            path_id=f"sim_{i:04d}",
            instruction=f"模拟指令_{i}"
        )
        results.append(result)

    metrics = evaluator.aggregate_metrics(results)

    return results, metrics


def main():
    """主函数 - 测试评估指标"""
    print("=" * 60)
    print("VLN 评估指标测试")
    print("=" * 60)

    # 运行模拟评估
    print("\n运行模拟评估 (100 样本)...")
    results, metrics = simulate_navigation_predictions(num_samples=100)

    # 打印报告
    print_evaluation_report(metrics, results)

    # 保存结果
    output_dir = Path("/Users/tyrion/Projects/Papers/data/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存指标
    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n指标已保存至：{metrics_file}")

    # 保存详细结果
    results_file = output_dir / "evaluation_results_detailed.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        detailed = []
        for r in results:
            detailed.append({
                'path_id': r.path_id,
                'instruction': r.instruction,
                'success': r.success,
                'trajectory_length': r.trajectory_length,
                'path_length': r.path_length,
                'distance_to_goal': r.distance_to_goal,
                'dtw_distance': r.dtw_distance,
                'oracle_success': r.oracle_success,
                'spl': r.spl
            })
        json.dump(detailed, f, indent=2)
    print(f"详细结果已保存至：{results_file}")

    print("\n" + "=" * 60)
    print("评估测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
