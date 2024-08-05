from openai import OpenAI
# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(api_key="sk-ab1f7c2b0847431fae89799456600f41", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个友好且善良的情感小助手。请尽力回复用户，帮助他们解决情感问题"},
        {"role": "user", "content": """
        您好！我最近真的很累。我公司中的事情太多了而且压力非常大。你能帮帮我吗？
        """},
  ],
    max_tokens=64,
    temperature=0.3,
    stream=False
)

print(response.choices[0].message.content)

#   "User": "Hey"
# },
# {
#   "AI Strategy": "",
#   "AI": "Hi there! How are you feeling today?"
# },
# {
#   "User": "Not great, to be honest. I had a big argument with a close friend, and now I'm feeling quite upset and confused."
# },
# {
#   "AI Strategy": "Emotional Validation",
#   "AI": "I'm sorry to hear that. Arguments with close friends can be really tough. It's understandable that you're feeling upset. Would you like to talk more about what happened?"
# },
# {
#   "User": "Thank you for understanding. We had a disagreement about a project we were working on together, and things got heated. Now we're not speaking to each other and it's weighing on me."
# },
# {
#   "AI Strategy": "Reflective Statements",
#   "AI": "It sounds like the argument has created a rift between both of you. That can be really challenging, especially if you value your friendship. How are you dealing with this situation at the moment?"
# },
#
#   '
#   对于上面的对话，请从“开心”“伤心”“恐惧”“悲伤””生气““厌恶”六种情绪中选择一种，并给激烈程度从1-10选择一个整数，数字越大表示情绪越激烈。
#   请按照这个输入格式输出：输出格式示例：“开心”，4