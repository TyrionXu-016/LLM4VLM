"""
采样 R2R 风格路径数据

用于中文 VLN 指令生成的测试数据
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../LLM4VLM

# R2R 风格场景列表（MatterPort3D）
SCENES = [
    "17DRP5sb8fy", "1LXtFkjw3qL", "1pXnuDYAj8r", "29hndJVuz6M",
    "2n8kARJN3HM", "2t7WUuJeko7", "2Z6YRPLGq5q", "5LpN3gCmAk5",
    "5q7pvUzZiPx", "5ZKStnWn8Zo", "8194nk5LbLH", "83J4ED9rt7f",
    "9a8jVREeNiE", "aayBHfsNo7d", "Ab74QMBbMDy", "b0c9b70e288",
    "bJTe4XMcgAV", "cJZExXp7XmJ", "d11c17a7c28d", "d7GhALbzg5i",
    "D7GhALbzg5i", "E823F0da4c2", "ecda6b5c73ad", "EDGbp7XiQpH",
    "EU6pjAgchCx", "f1Rt2ZbM2Jd", "fbnQ2eZfuzL", "fFHf7jRjSxH",
    "gTV8KGc7Nx7", "gXq7FMHf7jR", "Gv57R7EXJ2c", "gy86eM6cJ9f",
    "H15fVmFZ5z2", "H4c6V5ZgN6f", "hUzmcYgN5f2", "iRX9iYH5z2c",
    "J8kP5bHf7jR", "kB68hUzmcYg", "kMY2fRjSxHf", "LtD2ZbM2Jd8",
    "m8g7jRjSxHf", "mJd8LtD2ZbM", "n9hUzmcYgN5", "N5f2H15fVmF",
    "o68hUzmcYgN", "p7jRjSxHf1R", "q8LtD2ZbM2J", "r9mJd8LtD2Z",
]

# 场景类型映射
SCENE_TYPES = {
    "住宅": ["17DRP5sb8fy", "1LXtFkjw3qL", "2n8kARJN3HM", "5LpN3gCmAk5"],
    "办公室": ["8194nk5LbLH", "83J4ED9rt7f", "aayBHfsNo7d"],
    "酒店": ["2Z6YRPLGq5q", "5ZKStnWn8Zo", "D7GhALbzg5i"],
}

# 地标词库
LANDMARKS = [
    "沙发", "茶几", "餐桌", "椅子", "床", "床头柜", "衣柜", "书桌",
    "电视", "冰箱", "洗手台", "马桶", "淋浴间", "浴缸", "镜子",
    "楼梯", "电梯", "拱门", "走廊", "阳台", "窗户", "门", "地毯",
    "吊灯", "壁炉", "盆栽", "书架", "柜子", "装饰画", "花瓶",
    "前台", "接待处", "会议室", "休息区", "厨房", "卫生间", "卧室",
    "客厅", "餐厅", "玄关", "储藏室", "车库", "花园", "泳池",
]

# 起点/终点描述
START_LOCATIONS = [
    "入口处", "客厅门口", "玄关", "电梯口", "前台旁边", "大门",
    "走廊起点", "楼梯底部", "厨房门口", "卧室门口", "阳台入口",
]

END_LOCATIONS = [
    "窗边", "楼梯顶部", "走廊尽头", "沙发前", "餐桌旁",
    "床尾", "洗手台前", "阳台门口", "柜子旁边", "装饰画下方",
]


def generate_waypoints(num_waypoints: int = 4) -> list:
    """生成随机途经点"""
    return random.sample(LANDMARKS, min(num_waypoints, len(LANDMARKS)))


def generate_path(scene_id: str, path_id: int) -> dict:
    """生成单条路径信息"""
    # 确定场景类型
    scene_type = "室内"
    for st, scenes in SCENE_TYPES.items():
        if scene_id in scenes:
            scene_type = st
            break

    # 生成路径信息
    num_waypoints = random.randint(3, 6)
    distance = random.randint(8, 25)

    return {
        "path_id": f"path_{path_id:04d}",
        "scene_id": scene_id,
        "scene_type": scene_type,
        "start_location": random.choice(START_LOCATIONS),
        "waypoints": generate_waypoints(num_waypoints),
        "end_location": random.choice(END_LOCATIONS),
        "distance": distance,
        "num_variants": 3,  # 每个路径生成 3 个变体
    }


def sample_r2r_paths(num_paths: int = 10, output_file: str = None) -> list:
    """
    采样 R2R 风格路径

    Args:
        num_paths: 采样路径数量
        output_file: 输出文件路径

    Returns:
        路径信息列表
    """
    random.seed(42)  # 固定随机种子
    paths = []

    for i in range(num_paths):
        scene_id = random.choice(SCENES)
        path = generate_path(scene_id, i + 1)
        paths.append(path)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(paths, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(paths)} 条路径到：{output_file}")

    return paths


if __name__ == "__main__":
    # 生成测试路径
    paths = sample_r2r_paths(
        num_paths=10,
        output_file=str(REPO_ROOT / "data" / "sample_paths.json")
    )

    print("\n采样路径示例:")
    print("=" * 70)
    for path in paths[:3]:
        print(f"路径 ID: {path['path_id']}")
        print(f"  场景：{path['scene_type']} ({path['scene_id']})")
        print(f"  起点：{path['start_location']}")
        print(f"  途经：{' → '.join(path['waypoints'])}")
        print(f"  终点：{path['end_location']}")
        print(f"  距离：{path['distance']} 米")
        print()
