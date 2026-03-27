#!/usr/bin/env python3
"""
在 Habitat 模拟器中评估 VLN 模型性能

需要安装：
    pip install habitat-sim habitat-lab

需要配置 Matterport3D 数据集：
    https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import habitat
from habitat import Config, Env, VectorEnv, make_dataset, make_env
from habitat.tasks.nav.nav_task import NavigationTask
from habitat_baselines.common.obs_dict import ObservationDict

# 导入我们的 VLN 模型
import sys
sys.path.insert(0, '/Users/tyrion/Projects/Papers/code')
from vln_baseline_model import VLNBaseline, create_model


# ============================================================
# Habitat 配置
# ============================================================

def get_habitat_config() -> Config:
    """获取 Habitat 模拟器配置"""
    config = Config()

    # 模拟器配置
    config.SIMULATOR = Config()
    config.SIMULATOR.TYPE = "Sim-v0"
    config.SIMULATOR.AGENT_0 = Config()
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    config.SIMULATOR.RGB_SENSOR = Config()
    config.SIMULATOR.RGB_SENSOR.HEIGHT = 256
    config.SIMULATOR.RGB_SENSOR.WIDTH = 256
    config.SIMULATOR.RGB_SENSOR.HFOV = 90  # 视野角度

    # 任务配置
    config.TASK = Config()
    config.TASK.TYPE = "Nav-v0"
    config.TASK.SUCCESS_DISTANCE = 3.0  # 3 米内算成功
    config.TASK.GOAL_SENSOR_UUID = "goal"

    # 数据集配置
    config.DATASET = Config()
    config.DATASET.TYPE = "R2R"
    config.DATASET.DATA_PATH = "data/r2r/{split}/{split}.json.gz"
    config.DATASET.SCENES_DIR = "data/scene_datasets/"
    config.DATASET.SPLIT = "val_unseen"

    # 环境配置
    config.ENVIRONMENT = Config()
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 50
    config.ENVIRONMENT.ITERATOR_OPTIONS = Config()
    config.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False

    return config


# ============================================================
# VLN 评估器
# ============================================================

class VLNEvaluator:
    """VLN 模型评估器"""

    def __init__(self, model_path: str, config: Config):
        """
        Args:
            model_path: 模型检查点路径
            config: Habitat 配置
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.model = self._load_model(model_path)

        # 创建环境
        self.env = habitat.Env(config=config)

        # 统计信息
        self.episode_results = []

    def _load_model(self, model_path: str) -> VLNBaseline:
        """加载训练好的模型"""
        # TODO: 根据实际情况调整词汇表大小
        vocab_size = 5000
        model = create_model(vocab_size=vocab_size)

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"✓ 加载模型：{model_path}")
        return model

    def _extract_visual_features(self, observation: np.ndarray) -> torch.Tensor:
        """从观察中提取视觉特征（使用预训练的 ResNet-50）"""
        # TODO: 加载预训练的 ResNet-50
        # 这里简化为随机特征
        features = torch.randn(1, 36, 2048).to(self.device)
        return features

    def _tokenize_instruction(self, instruction: str) -> torch.Tensor:
        """将指令转换为 token IDs"""
        # TODO: 实现中文分词和词汇表映射
        # 这里简化为随机 token
        tokens = torch.randint(1, 5000, (1, 20)).to(self.device)
        return tokens

    def _get_candidate_directions(self, env) -> torch.Tensor:
        """获取候选方向的视觉特征"""
        # 获取 36 个候选方向的特征
        num_candidates = 36
        features = torch.randn(1, num_candidates, 256).to(self.device)
        return features

    def predict_action(self, instruction: str, observation: Dict) -> Tuple[int, float]:
        """
        预测下一个动作

        Returns:
            action_idx: 动作索引 (0-35)
            confidence: 置信度
        """
        # 准备输入
        tokens = self._tokenize_instruction(instruction)
        visual_features = self._extract_visual_features(observation)
        candidate_directions = self._get_candidate_directions(self.env)

        batch = {
            'instructions': tokens,
            'visual_features': visual_features,
            'candidate_directions': candidate_directions,
        }

        # 预测
        with torch.no_grad():
            action, confidence = self.model.predict(batch)

        return action.item(), confidence.item()

    def evaluate_episode(self, episode) -> Dict:
        """评估单个 episode"""
        self.env.reset()
        self.env.episode = episode

        instruction = episode.instruction.text
        positions = [self.env.sim.get_agent_state().position]

        done = False
        step = 0
        action_log = []

        while not done and step < 50:
            # 获取当前观察
            obs = self.env.sim.get_sensor_observations()

            # 预测动作
            action_idx, confidence = self.predict_action(instruction, obs)

            # 执行动作
            action = {
                'action': 'STOP' if action_idx == 0 else 'MOVE_FORWARD',
                'action_args': {}
            }

            obs, reward, done, info = self.env.step(action)

            positions.append(self.env.sim.get_agent_state().position)
            action_log.append({
                'step': step,
                'action': action_idx,
                'confidence': confidence
            })

            step += 1

        # 计算指标
        distance_to_goal = self.env.distance_to_goal()
        success = distance_to_goal < 3.0

        # 计算轨迹长度
        trajectory_length = 0
        for i in range(1, len(positions)):
            trajectory_length += np.linalg.norm(
                np.array(positions[i]) - np.array(positions[i-1])
            )

        # 计算 SPL
        reference_length = episode.info['geodesic_distance']
        if success:
            spl = min(trajectory_length, reference_length) / max(trajectory_length, reference_length)
        else:
            spl = 0.0

        return {
            'episode_id': episode.episode_id,
            'success': success,
            'spl': spl,
            'distance_to_goal': distance_to_goal,
            'trajectory_length': trajectory_length,
            'reference_length': reference_length,
            'steps': step,
            'action_log': action_log
        }

    def evaluate_all(self) -> Dict:
        """评估所有 episode"""
        results = []

        for episode in self.env.episode_iterator:
            result = self.evaluate_episode(episode)
            results.append(result)
            print(f"Episode {result['episode_id']}: "
                  f"Success={result['success']}, SPL={result['spl']:.3f}, "
                  f"Distance={result['distance_to_goal']:.2f}m")

        # 汇总统计
        metrics = {
            'num_episodes': len(results),
            'success_rate': np.mean([r['success'] for r in results]),
            'spl': np.mean([r['spl'] for r in results]),
            'avg_distance': np.mean([r['distance_to_goal'] for r in results]),
            'avg_steps': np.mean([r['steps'] for r in results]),
            'oracle_sr': np.mean([r['distance_to_goal'] < 3.0 for r in results]),
        }

        return metrics


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Habitat VLN 评估")
    print("=" * 60)

    # 配置
    config = get_habitat_config()
    model_path = "checkpoints/vln_baseline_best.pth"

    # 创建评估器
    evaluator = VLNEvaluator(model_path=model_path, config=config)

    # 评估
    metrics = evaluator.evaluate_all()

    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"Episode 数量：{metrics['num_episodes']}")
    print(f"成功率 (SR): {metrics['success_rate']:.2%}")
    print(f"SPL: {metrics['spl']:.4f}")
    print(f"平均距离：{metrics['avg_distance']:.2f}m")
    print(f"平均步数：{metrics['avg_steps']:.1f}")
    print(f"Oracle SR: {metrics['oracle_sr']:.2%}")

    # 保存结果
    output_path = "experiments/habitat_results.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ 结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
