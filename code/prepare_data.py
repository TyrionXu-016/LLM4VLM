"""
数据准备脚本

准备 R2R 增强数据用于训练：
1. 生成中文指令
2. 创建词表
3. 准备视觉特征
4. 生成训练/验证数据
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


def prepare_vocabulary():
    """准备中文词表"""
    print("准备词表...")

    # 基础字符集
    chars = [
        # 特殊标记
        '[PAD]', '[UNK]', '[CLS]', '[SEP]',
        # 方向词
        '左', '右', '前', '后', '直', '转', '走',
        # 位置词
        '上', '下', '进', '出', '过', '经', '到', '在',
        # 名词
        '米', '步', '楼', '梯', '走', '廊', '过', '道',
        '客', '厅', '卧', '室', '厨', '房', '卫', '生',
        '间', '门', '口', '窗', '户', '沙', '发', '桌',
        '椅', '床', '电', '梯', '冰', '箱', '餐', '具',
        # 数字
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '十', '百',
        # 标点
        '，', '。', '、',
    ]

    # 去重并创建映射
    unique_chars = list(dict.fromkeys(chars))
    char_to_id = {char: idx for idx, char in enumerate(unique_chars)}

    vocab_file = Path(__file__).parent.parent / 'data' / 'r2r_enhanced' / 'vocabulary.json'
    vocab_file.parent.mkdir(parents=True, exist_ok=True)

    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(char_to_id, f, ensure_ascii=False, indent=2)

    print(f"  ✓ 词表已保存：{vocab_file} ({len(char_to_id)} 字符)")
    return char_to_id


def prepare_sample_data():
    """准备样本数据用于测试"""
    print("准备样本数据...")

    sample_paths = [
        {
            "path_id": "sample_001",
            "path": ["起点", "客厅", "走廊", "厨房"],
            "landmarks": ["沙发", "餐桌", "楼梯"],
            "distance": 15
        },
        {
            "path_id": "sample_002",
            "path": ["门口", "走廊", "楼梯", "卧室"],
            "landmarks": ["椅子", "窗户", "床"],
            "distance": 20
        }
    ]

    sample_file = Path(__file__).parent.parent / 'data' / 'sample_paths.json'
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_paths, f, ensure_ascii=False, indent=2)

    print(f"  ✓ 样本数据已保存：{sample_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("数据准备脚本")
    print("=" * 60)

    prepare_vocabulary()
    prepare_sample_data()

    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
