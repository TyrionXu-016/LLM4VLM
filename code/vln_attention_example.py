"""
VLN 注意力机制简化示例
帮助理解 Anderson 2018 论文中的视觉 - 语言注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVLNAttention(nn.Module):
    """
    简化的 VLN 注意力机制

    输入:
    - instruction: 语言指令编码 [batch, seq_len, hidden]
    - views: 多个视角的图像特征 [batch, num_views, hidden]

    输出:
    - action_probs: 下一步动作概率 [batch, num_actions]
    """

    def __init__(self, hidden_size=512, num_actions=6):
        super().__init__()
        self.hidden_size = hidden_size

        # 语言编码器 (简化为线性层)
        self.language_encoder = nn.LSTM(
            input_size=300,  # GloVe 词向量维度
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # 视觉编码器 (简化为线性层)
        self.visual_encoder = nn.Linear(2048, hidden_size)  # ResNet 特征

        # 注意力机制
        self.attention = nn.Linear(hidden_size * 2, 1)

        # 动作预测
        self.action_head = nn.Linear(hidden_size * 2, num_actions)

    def forward(self, instruction_words, view_features):
        """
        前向传播

        Args:
            instruction_words: [batch, seq_len, 300] 词向量
            view_features: [batch, num_views, 2048] 图像特征
        """
        batch_size, seq_len, _ = instruction_words.shape
        num_views = view_features.shape[1]

        # Step 1: 编码语言指令
        # [batch, seq_len, hidden]
        lang_out, (lang_hidden, _) = self.language_encoder(instruction_words)
        # 取最后一个时间步作为句子表示
        lang_features = lang_out[:, -1, :]  # [batch, hidden]

        # Step 2: 编码视觉特征
        # [batch, num_views, hidden]
        vis_features = self.visual_encoder(view_features)
        vis_features = F.relu(vis_features)

        # Step 3: 计算注意力
        # 将语言特征广播到每个视角
        lang_expanded = lang_features.unsqueeze(1).expand(-1, num_views, -1)

        # 拼接语言和视觉特征
        combined = torch.cat([lang_expanded, vis_features], dim=-1)  # [batch, num_views, hidden*2]

        # 计算注意力分数
        attention_scores = self.attention(combined).squeeze(-1)  # [batch, num_views]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, num_views]

        # 加权平均视觉特征
        vis_context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch, 1, num_views]
            vis_features  # [batch, num_views, hidden]
        ).squeeze(1)  # [batch, hidden]

        # Step 4: 预测动作
        final_features = torch.cat([lang_features, vis_context], dim=-1)
        action_logits = self.action_head(final_features)
        action_probs = F.softmax(action_logits, dim=-1)

        return action_probs, attention_weights


# === 使用示例 ===
if __name__ == "__main__":
    # 模拟输入
    batch_size = 4
    seq_len = 20  # 指令长度 (词)
    num_views = 36  # 每个位置的视角数 (6 个水平 × 3 个垂直 + 1 个仰视)

    model = SimpleVLNAttention(hidden_size=512, num_actions=6)

    # 随机输入
    instruction = torch.randn(batch_size, seq_len, 300)  # 词向量
    views = torch.randn(batch_size, num_views, 2048)  # ResNet 特征

    # 前向传播
    action_probs, attention_weights = model(instruction, views)

    print("输入指令形状:", instruction.shape)
    print("输入视角形状:", views.shape)
    print("输出动作概率形状:", action_probs.shape)
    print("注意力权重形状:", attention_weights.shape)
    print("\n注意力权重 (每个视角的权重):")
    print(attention_weights)
    print("\n动作概率 (6 个动作):")
    print(action_probs)

    # 解释
    print("\n=== 结果解释 ===")
    for i in range(batch_size):
        best_view = attention_weights[i].argmax().item()
        best_action = action_probs[i].argmax().item()
        print(f"样本 {i}: 最关注的视角是 #{best_view}, 最可能的动作是 #{best_action}")

    print("\n=== 动作含义 ===")
    print("动作索引: 0=停止，1=前进，2=左转，3=右转，4=向上看，5=向下看")
