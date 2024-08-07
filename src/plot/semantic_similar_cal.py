from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 加载预训练的word2vec模型
model = KeyedVectors.load_word2vec_format("G:\VRLLM\GoogleNews-vectors-negative300.bin", binary=True)

# 定义三个阶段
stages = [
    "Exploration: Ask open-ended questions, listen, and try to understand the other person's emotions and issues. Avoid rushing to provide solutions; instead, encourage dialogue to help the other person better understand their emotions and challenges.",
    "Comforting: Express empathy, comfort the other person through affectionate language and an affirming attitude. Provide support and encouragement to make the other person feel understood and accepted.",
    "Action: Offer practical advice and solutions, help the other person think and formulate action plans. Encourage proactive actions and provide support and feedback to ensure that the problem is resolved."
]

# 定义策略和其行为描述（去掉示例部分）
strategies = {
    "Reflective Statements (RS)": "Repeat or rephrase what the User has expressed to show that you’re actively listening.",
    "Clarification (Cla)": "Seek clarification to ensure a clear understanding of the User’s emotions and experiences.",
    "Emotional Validation (EV)": "Acknowledge and validate the User’s emotions without judgment.",
    "Empathetic Statements (ES)": "Express understanding and empathy towards the User’s experiences.",
    "Affirmation (Aff)": "Provide positive reinforcement and encouragement to uplift the User’s spirits.",
    "Offer Hope (OH)": "Share optimistic perspectives or possibilities to instill hope.",
    "Avoid Judgment and Criticism (AJC)": "Create a non-judgmental and safe space for the User to express their emotions without fear of criticism.",
    "Suggest Options (SO)": "Offer practical suggestions or alternative perspectives for addressing the issue at hand.",
    "Collaborative Planning (CP)": "Work together with the User to develop an action plan.",
    "Provide Different Perspectives (PDP)": "Offer alternative ways of looking at the situation to help the User gain new insights.",
    "Reframe Negative Thoughts (RNT)": "Help the User reframe negative thoughts into more positive or realistic ones.",
    "Share Information (SI)": "Provide educational or factual information about emotions, coping mechanisms, or self-care practices.",
    "Normalize Experiences (NE)": "Explain that certain emotions or reactions are common and part of the human experience.",
    "Promote Self-Care Practices (PSP)": "Advocate for engaging in activities that promote well-being and self-care.",
    "Stress Management (SM)": "Provide suggestions for stress management techniques like exercise, meditation, or spending time in nature."
}

# strategies = {
#     "Reflective Statements (RS)": "Repeat or rephrase what the User has expressed to show that you’re actively listening. Example: User: “I’m feeling really overwhelmed with all the work I have to do.” Assistant: “It sounds like you’re feeling overwhelmed with your workload.”",
#     "Clarification (Cla)": "Seek clarification to ensure a clear understanding of the User’s emotions and experiences. Example: User: “I just can’t shake off this feeling of sadness.” Assistant: “Could you help me understand what might have triggered this feeling of sadness?”",
#     "Emotional Validation (EV)": "Acknowledge and validate the User’s emotions without judgment. Example: User: “I’m so frustrated with myself for making the same mistake again.” Assistant: “It’s completely understandable to feel frustrated when you make a mistake.”",
#     "Empathetic Statements (ES)": "Express understanding and empathy towards the User’s experiences. Example: User: “I’m really struggling with my self-confidence right now.” Assistant: “I can imagine how challenging it must be to navigate through situations that affect your self-confidence.”",
#     "Affirmation (Aff)": "Provide positive reinforcement and encouragement to uplift the User’s spirits. Example: User: “I feel like I’m not good enough.” Assistant: “You’ve accomplished so much already, and your abilities speak for themselves. Don’t underestimate your capabilities.”",
#     "Offer Hope (OH)": "Share optimistic perspectives or possibilities to instill hope. Example: User: “I don’t know if things will ever get better.” Assistant: “Remember that change is constant, and there are always opportunities for growth and positive change.”",
#     "Avoid Judgment and Criticism (AJC)": "It’s important to create a non-judgmental and safe space for the User to express their emotions without fear of criticism. Refrain from passing judgment or being overly critical of their experiences or choices. Example: User: “I’m feeling so guilty for taking time off work to focus on my mental health.” Assistant: “Taking care of your mental health is crucial, and it’s not something to feel guilty about. Your well-being should always be a priority, and I’m glad you recognized that. Is there anything I can do to support you during this time?”",
#     "Suggest Options (SO)": "Offer practical suggestions or alternative perspectives for addressing the issue at hand. Example: User: “I’m having trouble managing my stress.” Assistant: “Have you considered trying relaxation techniques like deep breathing or mindfulness exercises?”",
#     "Collaborative Planning (CP)": "Work together with the User to develop an action plan. Example: User: “I want to improve my time management skills.” Assistant: “Let’s brainstorm some strategies together. How about breaking tasks into smaller, more manageable chunks?”",
#     "Provide Different Perspectives (PDP)": "Offer alternative ways of looking at the situation to help the User gain new insights. Example: User: “I’m devastated that my project didn’t succeed.” Assistant: “Sometimes setbacks can lead to unexpected opportunities for learning and growth. It’s a chance to reassess and try again.”",
#     "Reframe Negative Thoughts (RNT)": "Help the User reframe negative thoughts into more positive or realistic ones. Example: User: “I’m such a failure.” Assistant: “Instead of thinking that way, let’s focus on what you’ve learned from this experience and how you can apply it moving forward.”",
#     "Share Information (SI)": "Provide educational or factual information about emotions, coping mechanisms, or self-care practices. Example: User: “I’m struggling to manage my anxiety.” Assistant: “Did you know that deep breathing exercises and grounding techniques can help reduce anxiety symptoms? Would you like me to explain how to practice them?”",
#     "Normalize Experiences (NE)": "Explain that certain emotions or reactions are common and part of the human experience. Example: User: “I feel so guilty for taking time for myself.” Assistant: “It’s common to feel guilty about self-care, but it’s essential for your well-being. Remember, you deserve to prioritize your needs too.”",
#     "Promote Self-Care Practices (PSP)": "Advocate for engaging in activities that promote well-being and self-care. Example: “Make sure to take some time for yourself and do something that brings you joy and relaxation.”",
#     "Stress Management (SM)": "Provide suggestions for stress management techniques like exercise, meditation, or spending time in nature. Example: “Engaging in regular physical activity can help reduce stress and improve mood.”"
# }

# 定义一个函数来计算两个句子的语义相似度
def compute_similarity(sentence1, sentence2, model):
    tokens1 = [word for word in sentence1.lower().split() if word in model]
    tokens2 = [word for word in sentence2.lower().split() if word in model]

    if not tokens1 or not tokens2:
        return 0.0

    vector1 = np.mean([model[word] for word in tokens1], axis=0)
    vector2 = np.mean([model[word] for word in tokens2], axis=0)

    return cosine_similarity([vector1], [vector2])[0][0]

# 计算每个策略和三个阶段之间的相似度
similarity_scores = {}
for strategy, description in strategies.items():
    similarity_scores[strategy] = [compute_similarity(description, stage, model) for stage in stages]

# 打印相似度分数
for strategy, scores in similarity_scores.items():
    print(f"{strategy} - Exploration: {scores[0]:.4f}, Comforting: {scores[1]:.4f}, Action: {scores[2]:.4f}")
