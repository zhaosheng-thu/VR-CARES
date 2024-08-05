import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import warnings
# 关闭所有警告
warnings.filterwarnings("ignore")
import re


def remove_incomplete_sentence(text):
    # 使用正则表达式找到最后一个句子结束标点（这里假设句子结束标点为'.'、'!'或'?'）
    match = re.search(r'([.!?])[^.!?]*$', text)
    if match:
        # 如果找到了句子结束标点，那么返回从文本开始到该标点的部分
        return text[:match.start(1) + 1]
    else:
        # 如果没有找到句子结束标点，那么返回整个文本
        return text

# 测试函数
# text = "This is a complete sentence. This is an incomplete sentence"
# print(remove_incomplete_sentence(text))


device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained('/root/szhao/model-weights/Llama-3-8B',
                                             device_map=device_map, torch_dtype=torch.float16)

# if it is finetuned with litgpt, try the following code
state_dict = torch.load('/root/szhao/ES-Lora/litgpt/litgpt/out/convert/hf-llama3-instruct-esconv/model.pth')
model.load_state_dict(state_dict)

model = model.eval()

tokenizer = AutoTokenizer.from_pretrained('/root/szhao/model-weights/Llama-3-8B-Instruct')
tokenizer.pad_token = tokenizer.eos_token

# init the dialogue with the background tokens
background_dialogue = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {{ You are a helpful assistant"
        }}'''
        
text_starts = background_dialogue
# start the conversation
rounds = 5

dialogue = "".join([text_starts, '<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n'])

for i in range(rounds): 
    print("Round: ", i + 1)
    new_input = input("You: ")
    
    dialogue += f'{{{{{ {new_input} }}}}}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n'
    # 将对话转换为模型可以理解的输入
    input_ids = tokenizer([dialogue], return_tensors="pt", add_special_tokens=False).input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')

    generate_input = {
        "input_ids":input_ids,
        # max token num is set to 128, it controls the length of the response
        "max_new_tokens":128,
        "do_sample":True,
        "top_k":50,
        "top_p":0.95,
        "temperature":0.3,
        "repetition_penalty":1.3,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.pad_token_id
    }
    generate_ids = model.generate(**generate_input)
    text = tokenizer.decode(generate_ids[0])
    print("text", text)
    output = text.split("assistant")[-1].replace("<|start_header_id|>", "").replace("<|end_header_id|>", "").strip()
    # 防止输出中包含一轮内生成多余的'user'
    if 'user' in output:
        output = output.split('user')[0]
    # 截取说完的话
    output = "".join([remove_incomplete_sentence(output), ''])
    print(f"\nAssistant: {output}\n")
    # 更新对话，准备下一轮
    dialogue += f'{{{{{ output }}}}}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n'

