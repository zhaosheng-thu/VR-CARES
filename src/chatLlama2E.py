import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
# 关闭所有警告
warnings.filterwarnings("ignore")

device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained('/root/szhao/model-weights/Llama-2-7b-chat-hf',
                                             device_map=device_map, torch_dtype=torch.float16)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained('/root/szhao/model-weights/Llama-2-7b-chat-hf')
tokenizer.pad_token = tokenizer.eos_token

# init the dialogue with the background tokens
background_dialogue = '''<s>Human: You are an Assistant, designed for emotional support task. 
                Now please imitate as a real human and chat with me, who is a person needs emotional support.
                </s>\n<s>Assistant:'''
background_tokens = tokenizer(background_dialogue, return_tensors="pt").input_ids
if torch.cuda.is_available():
    background_tokens = background_tokens.to('cuda')

generate_starts = model.generate(input_ids=background_tokens, max_new_tokens=128)
text_starts = tokenizer.decode(generate_starts[0])
print("text_starts: ", text_starts)

# start the conversation
rounds = 5
# text_starts = ""
dialogue = text_starts + '<s>Human:'

for i in range(rounds): 
    print("Round: ", i + 1)
    new_input = input("You: ")
    
    dialogue += new_input + '</s>\n<s>Assistant: '
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
    output = text.split("Assistant: ")[-1].replace("<s>", "").replace("</s>", "").strip()
    # 防止输出中包含一轮内生成多余的'Human'
    if 'Human' in output:
        output = output.split('Human')[0]
    # 截取说完的话
    output = output.rsplit('.', 1)[0] + '.'
    print(f"\nAssistant: {output}\n")
    # 更新对话，准备下一轮
    dialogue += f'{output}</s>\n<s>Human: '

