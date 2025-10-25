import pandas as pd
import json

# 读取数据
df = pd.read_csv('Synthetic-Persona-Chat_test.csv')

# 构建训练样本
training_samples = []

for idx, row in df.iterrows():
    conversation = row['Best Generated Conversation']

    # 解析对话轮次
    turns = conversation.split('\n')
    dialogue_history = []

    for i, turn in enumerate(turns):
        if turn.startswith('User 1:'):
            user_input = turn.replace('User 1:', '').strip()
            if i + 1 < len(turns) and turns[i + 1].startswith('User 2:'):
                assistant_response = turns[i + 1].replace('User 2:', '').strip()

                training_sample = {
                    'instruction': user_input,
                    'input': '\n'.join(dialogue_history) if dialogue_history else '',
                    'output': assistant_response,
                    'personas': row['user 1 personas']  # 使用用户1的人物设定
                }
                training_samples.append(training_sample)

        elif turn.startswith('User 2:'):
            user_input = turn.replace('User 2:', '').strip()
            # 类似的，为另一个用户的回复准备数据
            if i + 1 < len(turns) and turns[i + 1].startswith('User 1:'):
                assistant_response = turns[i + 1].replace('User 1:', '').strip()

                training_sample = {
                    'instruction': user_input,
                    'input': '\n'.join(dialogue_history) if dialogue_history else '',
                    'output': assistant_response,
                    'personas': row['user 2 personas']  # 使用用户2的人物设定
                }
                training_samples.append(training_sample)

        # 更新对话历史
        dialogue_history.append(turn.strip())