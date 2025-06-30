from openai import OpenAI
client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://0.0.0.0:23333/v1"
)
model_name = client.models.list().data[0].id
print(model_name)\

reward_template = """
你是一个模型评估专家。你的任务是根据给定的标准答案和模型生成的回答，评估回答的质量。请根据以下标准进行评分:
1. **内容相关性**: 回答是否与提示内容相关？
2. **准确性**: 回答是否准确？是否包含错误信息？
3. **完整性**: 回答是否全面？是否遗漏了重要信息？
4. **清晰度**: 回答是否清晰易懂？是否有语法错误或拼写错误？
请根据以上标准对以下回答进行评分, 评分范围从0到10, 0表示非常差, 10表示非常好. 请给出评分.
标准答案: {reference}
模型回答: {response}

要求: 直接返回分数, 不要返回其他内容。
""".format(reference="你好", response="世界")


response = client.chat.completions.create(
    model=model_name,
    messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": reward_template},
    ],
    temperature=0.8,
    top_p=0.8
)
print(response)
print(response.choices[0].message.content)
print("====")
reward_answer = response.choices[0].message.content
try:
    score = float(reward_answer.split("</think>")[-1].strip())  # 假设分数在回答的第一部分
except ValueError:
    score = 0.0
print(score)
