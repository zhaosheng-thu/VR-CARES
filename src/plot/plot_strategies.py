import matplotlib.pyplot as plt
import numpy as np

# 策略列表
strategies = [
    "Reflective Statements (RS)", "Clarification (Cla)", "Emotional Validation (EV)",
    "Empathetic Statements (ES)", "Affirmation (Aff)", "Offer Hope (OH)",
    "Avoid Judgment and Criticism (AJC)", "Suggest Options (SO)", "Collaborative Planning (CP)",
    "Provide Different Perspectives (PDP)", "Reframe Negative Thoughts (RNT)",
    "Share Information (SI)", "Normalize Experiences (NE)", "Promote Self-Care Practices (PSP)",
    "Stress Management (SM)"
]

# 各阶段的分数
exploration_scores = [0.6342, 0.7515, 0.7105, 0.6046, 0.4818, 0.5634, 0.7351, 0.5851, 0.5401, 0.6836, 0.6869, 0.5596, 0.7565, 0.5415, 0.6424]
comforting_scores = [0.6427, 0.6344, 0.6088, 0.7109, 0.6352, 0.5449, 0.7079, 0.5896, 0.5639, 0.6222, 0.5962, 0.5843, 0.6755, 0.5730, 0.5943]
action_scores = [0.6251, 0.6841, 0.4851, 0.5221, 0.6452, 0.5599, 0.5807, 0.7428, 0.6620, 0.7384, 0.6496, 0.6101, 0.6012, 0.6178, 0.6947]

# 设置图表的大小和样式
fig, ax = plt.subplots(figsize=(12, 10))
bar_width = 0.25
index = np.arange(len(strategies))

# 绘制每个阶段的条形图
bars1 = ax.bar(index - bar_width, exploration_scores, bar_width, label='Exploration')
bars2 = ax.bar(index, comforting_scores, bar_width, label='Comforting')
bars3 = ax.bar(index + bar_width, action_scores, bar_width, label='Action')

# 添加标签和标题
ax.set_xlabel('Strategies')
ax.set_ylabel('Scores')
ax.set_title('Scores for Each Strategy in Different Stages')
ax.set_xticks(index)
ax.set_xticklabels(strategies, rotation=90)
ax.legend()

# 添加数据标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.tight_layout()
plt.show()

##norm
import matplotlib.pyplot as plt
import numpy as np

# 策略列表
strategies = [
    "Reflective Statements (RS)", "Clarification (Cla)", "Emotional Validation (EV)",
    "Empathetic Statements (ES)", "Affirmation (Aff)", "Offer Hope (OH)",
    "Avoid Judgment and Criticism (AJC)", "Suggest Options (SO)", "Collaborative Planning (CP)",
    "Provide Different Perspectives (PDP)", "Reframe Negative Thoughts (RNT)",
    "Share Information (SI)", "Normalize Experiences (NE)", "Promote Self-Care Practices (PSP)",
    "Stress Management (SM)"
]

# 各阶段的分数
exploration_scores = np.array([0.6342, 0.7515, 0.7105, 0.6046, 0.4818, 0.5634, 0.7351, 0.5851, 0.5401, 0.6836, 0.6869, 0.5596, 0.7263, 0.5415, 0.6424])
comforting_scores = np.array([0.6427, 0.6344, 0.6088, 0.7109, 0.6352, 0.5449, 0.7079, 0.5896, 0.5639, 0.6222, 0.5962, 0.5843, 0.6944, 0.5730, 0.5943])
action_scores = np.array([0.6251, 0.6841, 0.4851, 0.5221, 0.6452, 0.5599, 0.5807, 0.7428, 0.6620, 0.7384, 0.6496, 0.6101, 0.6102, 0.6178, 0.6947])

# 组合各个策略在三个阶段的分数
scores = np.stack([exploration_scores, comforting_scores, action_scores], axis=1)

# 对每个策略的三个阶段的分数进行组内标准化
scores_norm = (scores - np.mean(scores, axis=1, keepdims=True)) / np.std(scores, axis=1, keepdims=True)

# 设置图表的大小和样式
fig, ax = plt.subplots(figsize=(12, 10))
bar_width = 0.3
index = np.arange(len(strategies))

# 绘制每个阶段的条形图
bars1 = ax.bar(index - bar_width, scores_norm[:, 0], bar_width, label='Exploration', color='lightblue')
bars2 = ax.bar(index, scores_norm[:, 1], bar_width, label='Comforting', color='lightgreen')
bars3 = ax.bar(index + bar_width, scores_norm[:, 2], bar_width, label='Action', color='salmon')

# 添加标签和标题
ax.set_xlabel('Strategies')
ax.set_ylabel('Standardized Scores')
ax.set_title('Standardized Semantic Similarity of Strategies with Different Stages')
ax.set_xticks(index)
ax.set_xticklabels(strategies, rotation=90)
ax.legend()

# 添加数据标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.tight_layout()
plt.show()
