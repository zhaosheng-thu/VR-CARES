import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
# 关闭所有警告
warnings.filterwarnings("ignore")

device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained('/root/szhao/model-weights/Llama2-Chinese-7b-Chat',
                                             device_map=device_map, torch_dtype=torch.float16)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained('/root/szhao/model-weights/Llama2-Chinese-7b-Chat')
tokenizer.pad_token = tokenizer.eos_token


rounds = 7
dialogue = '<s>Human:'

for i in range(rounds): 
    print("Round: ", i + 1)
    new_input = input("You: ")
    
    dialogue += new_input + '\n</s><s>Assistant: '
    # 将对话转换为模型可以理解的输入
    input_ids = tokenizer([dialogue], return_tensors="pt", add_special_tokens=False).input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')

    generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":512,
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
    print("Assistant: ", text.split("Assistant: ")[-1].replace("<s>", "").replace("</s>", "").strip())
    # 更新对话，准备下一轮
    dialogue = text + '\n</s><s>Human: '
    # print("\ndialogue: ", dialogue)
    print("\n")
