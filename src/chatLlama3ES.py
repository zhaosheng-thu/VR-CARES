import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import warnings
# 关闭所有警告
warnings.filterwarnings("ignore")
import re
import time
import os
# print(torch.cuda.is_available())

TEMPLATE_ASSISTANT = '<|start_header_id|>Assistant<|end_header_id|>'
TEMPLATE_ASSISTANT_LOWER = '<|start_header_id|>assistant<|end_header_id|>'
TEMPLATE_ASSISTANT_STRATEGY = '<|start_header_id|>Assistant Strategy<|end_header_id|>'
TEMPLATE_USER = '<|start_header_id|>User<|end_header_id|>'
TEMPLATE_SYS = '<|start_header_id|>system<|end_header_id|>'
TEMPLATE_BACKGROUND = "<|start_header_id|>User's background<|end_header_id|>"
OUTPUT_DIR = '/root/szhao/ES-Lora/litgpt/litgpt/out_assistant'
INPUT_DIR = '/root/szhao/ES-Lora/litgpt/litgpt/in_assistant'


def remove_incomplete_sentence(text):
    # 使用正则表达式找到最后一个句子结束标点（这里假设句子结束标点为'.'、'!'或'?'）
    match = re.search(r'([.!?])[^.!?]*$', text)
    if match:
        # 如果找到了句子结束标点，那么返回从文本开始到该标点的部分
        return text[:match.start(1) + 1]
    else:
        # 如果没有找到句子结束标点，那么返回整个文本
        return text


def find_substring_rank(text, substring):
    # 找到子串在文本中的所有位置
    positions = []
    position = text.find(substring)
    while position != -1:
        positions.append(position)
        position = text.find(substring, position + 1)
    return positions


def os_output(text: str=None, round: int=-1):
    if round == -1:
        files = os.listdir(OUTPUT_DIR)
        for file in files:
        # 获取文件的完整路径
            file_path = os.path.join(OUTPUT_DIR, file)
            if file.endswith('.txt'):
                os.remove(file_path)
    # else 写入
    else:
        with open(f'{OUTPUT_DIR}/output_{round}.txt', 'w') as f:
            f.write(text)


def os_input(round: int=-1):
    if round == -1:
        files = os.listdir(INPUT_DIR)
        for file in files:
        # 获取文件的完整路径
            file_path = os.path.join(INPUT_DIR, file)
            if file.endswith('.txt'):
                os.remove(file_path)
        return None
    # else 读取
    else:
        while True:
            time.sleep(1)
            if os.path.exists(f'{INPUT_DIR}/input_{round}.txt'):
                break
        with open(f'{INPUT_DIR}/input_{round}.txt', 'r') as f:
            text = f.read()
        return text


def find_role_from_strategy(strategy, round):
    strategy_role = {
    "Reflective Statements": 1,
    "Clarification": 1,
    "Emotional Validation": 1,
    "Empathetic Statements": 2,
    "Affirmation": 2,
    "Offer Hope": 3,
    "Avoid Judgment And Criticism": 1,
    "Suggest Options": 3,
    "Collaborative Planning": 3,
    "Provide Different Perspectives": 2,
    "Reframe Negative Thoughts": 3,
    "Share Information": 3,
    "Normalize Experiences": 2,
    "Promote Self-Care Practices": 3,
    "Stress Management": 3,
    # "Others": 0
}
    return strategy_role[strategy] if strategy in strategy_role else 1 if round < 3 else 3



def main():
    # set all the process on a single gpu
    device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained('/root/szhao/model-weights/Llama-3-8B-Instruct',
                                                 device_map=device_map, torch_dtype=torch.float16)

    # if it is finetuned with litgpt, try the following code
    state_dict = torch.load('litgpt/litgpt/out/convert/hf-llama3-instruct-esconv/model.pth')
    model.load_state_dict(state_dict)

    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained('litgpt/litgpt/checkpoints/meta-llama/Llama-3-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token

    # init the dialogue with the background<|start_header_id|>system<|end_header_id|> tokens
    system_prompt = f'<|begin_of_text|>{TEMPLATE_SYS}\n\nYou are a helpful assistant designed for emotion support task. Now there is a scene.<|eot_id|>\n' 
    background_prompt = f'{TEMPLATE_BACKGROUND}\n\n<|eot_id|>\n'
    text_starts = f'{system_prompt}{background_prompt}'        

    print("text_starts: ", text_starts)

    # start the conversation
    rounds = 20
    # text_starts = ""
    dialogue = "".join([text_starts, f'{TEMPLATE_USER}\n\n'])
    generate_kwargs = {
        "input_ids":None,
        # max token num is set to 128, it controls the length of the response
        "max_new_tokens":32,
        "do_sample":True,
        "top_k":50,
        "top_p":0.95,
        "temperature":0.3,
        "repetition_penalty":1.3,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.pad_token_id
    }
    # 清空文件夹
    os_input()
    os_output()
    
    
    for i in range(-1, rounds): 
        
        print("Round: ", i + 1)
        print(f"\nMemory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
        if i == -1:
            new_input = "Hello, Can you help me?"
        else:
            # new_input = input("You: ")
            new_input = os_input(round=i)
        dialogue += f'{new_input}<|eot_id|>\n{TEMPLATE_ASSISTANT_STRATEGY}\n\n'
        # 生成 Strategy 
        input_ids = tokenizer([dialogue], return_tensors="pt", add_special_tokens=False).input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
            
        generate_kwargs["input_ids"] = input_ids
        generate_ids = model.generate(**generate_kwargs)
        text = tokenizer.decode(generate_ids[0])
        output_new = text.split(f"{TEMPLATE_USER}")[i + 2].replace(TEMPLATE_ASSISTANT_LOWER, TEMPLATE_ASSISTANT)
        # 截取 AI Stratrgy 部分
        start_output_strategy = find_substring_rank(output_new, TEMPLATE_ASSISTANT_STRATEGY)[0] + len(TEMPLATE_ASSISTANT_STRATEGY) + len('\n\n')
        end_output_strategy = find_substring_rank(output_new[start_output_strategy: ], "<|eot_id|>")[0] + start_output_strategy
        # print("start-end", start_output_strategy, end_output_strategy)
        output_strategy = output_new[start_output_strategy: end_output_strategy]
        dialogue += f'{output_strategy}<|eot_id|>\n{TEMPLATE_ASSISTANT}\n\n'
        
        # 生成 Assistant 对话
        input_ids = tokenizer([dialogue], return_tensors="pt", add_special_tokens=False).input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
          
        generate_kwargs["input_ids"] = input_ids  
        generate_kwargs["max_new_tokens"] = 96
        generate_kwargs["temperature"] = 0.7
        generate_ids = model.generate(**generate_kwargs)
        text = tokenizer.decode(generate_ids[0])
        output_new = text.split(f"{TEMPLATE_ASSISTANT_STRATEGY}")[i + 2].replace(TEMPLATE_ASSISTANT_LOWER, TEMPLATE_ASSISTANT)
        
        # 截取 AI Assistant 部分
        start_output_assistant = find_substring_rank(output_new, TEMPLATE_ASSISTANT)[0] + len(TEMPLATE_ASSISTANT) + len('\n\n')
        eot_id_pos_list = find_substring_rank(output_new[start_output_assistant: ], '<|eot_id|>')
        end_output_assistant = eot_id_pos_list[0] + start_output_assistant if len(eot_id_pos_list) > 0 else len(output_new[start_output_assistant: ]) + start_output_assistant
        output_assistant = output_new[start_output_assistant: end_output_assistant] if len(eot_id_pos_list) > 0 else remove_incomplete_sentence(output_new[start_output_assistant: end_output_assistant])
        
        dialogue += f'{output_assistant}<|eot_id|>\n{TEMPLATE_USER}\n\n'
        print(f"\nAssistant Strategy: {output_strategy}\n")
        print(f"Assistant: {output_assistant}\n")
        
        output_user_prompt = "Optional Prompt"
        ##### Optional #####
        # generate the optional prompt for the participants
        # input_ids = tokenizer([dialogue], return_tensors="pt", add_special_tokens=False).input_ids
        # if torch.cuda.is_available():
        #     input_ids = input_ids.to('cuda')
            
        # generate_kwargs["max_new_tokens"] = 128
        # generate_kwargs["input_ids"] = input_ids
        # generate_ids = model.generate(**generate_kwargs)
        # text = tokenizer.decode(generate_ids[0])
        # output_new = text.split(f"{TEMPLATE_ASSISTANT}")[i + 1]
        
        # start_output_user_prompt = find_substring_rank(output_new, TEMPLATE_USER)[0] + len(TEMPLATE_USER) + len('\n\n')
        # end_output_user_prompt = find_substring_rank(output_new[start_output_user_prompt: ], '<|eot_id|>')[0] + start_output_user_prompt
        # # print("start-end", start_output_assistant, end_output_assistant)
        # output_user_prompt = output_new[start_output_user_prompt: end_output_user_prompt]
        # print("Output User prompt(Option): ", output_user_prompt)
        ##########
        if i != -1:
            os_output(text=f"{output_assistant}***{output_strategy}_{find_role_from_strategy(output_strategy, i)}***{output_user_prompt}", round=i) 


if __name__ == '__main__':
    main()