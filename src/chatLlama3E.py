import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import warnings
# 关闭所有警告
warnings.filterwarnings("ignore")
import re


def remove_incomplete_sentence(text):
    # re
    match = re.search(r'([.!?])[^.!?]*$', text)
    if match:
        return text[:match.start(1) + 1]
    else:
        return text


device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained('Llama-3-8B',
                                             device_map=device_map, torch_dtype=torch.float16)

# if it is finetuned with litgpt, try the following code
state_dict = torch.load('litgpt/out/convert/hf-llama3-instruct-esconv/model.pth')
model.load_state_dict(state_dict)

model = model.eval()

tokenizer = AutoTokenizer.from_pretrained('Llama-3-8B-Instruct')
tokenizer.pad_token = tokenizer.eos_token

# init the dialogue with the background tokens
background_dialogue = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {{ You are a helpful assistant designed for emotion support task. "
        "Now there is a scene }}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {{ Hello, can you help me? }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

background_tokens = tokenizer(background_dialogue, return_tensors="pt").input_ids
print("background_tokens: ", background_tokens)
if torch.cuda.is_available():
    background_tokens = background_tokens.to('cuda')

generate_starts = model.generate(input_ids=background_tokens, max_new_tokens=64, eos_token_id=tokenizer.eos_token_id,)
text_starts = tokenizer.decode(generate_starts[0])
print("text_starts: ", text_starts)

# start the conversation
rounds = 20
# text_starts = ""
dialogue = "".join([text_starts, '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'])

for i in range(rounds): 
    print("Round: ", i + 1)
    new_input = input("You: ")
    
    dialogue += f'{{{{{ {new_input} }}}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    input_ids = tokenizer([dialogue], return_tensors="pt", add_special_tokens=False).input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')

    generate_input = {
        "input_ids":input_ids,
        # max token num is set to **, it controls the length of the response
        "max_new_tokens":64,
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
    if 'user' in output:
        output = output.split('user')[0]
    output = "".join([remove_incomplete_sentence(output), ''])
    print(f"\nAssistant: {output}\n")
    # update
    dialogue += f'{{{{{ output }}}}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'

