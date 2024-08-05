# model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatModel:
    def __init__(self, root_dir_to_model):
        self.rot_dir_to_model = root_dir_to_model
        self.model_path = f'{root_dir_to_model}'
        self.model = self.load_model(self.model_path)
        self.tokenizer = self.load_tokenizer(self.model_path)

    def load_model(self, model_path):
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                     device_map=device_map)
        model.eval()
        return model

    def load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # The initial dialogue for llama chatbox
    def init_dialogue(self):
        return '<s>Human: You are an Assistant, designed for emotional support task. ' \
               'Now please imitate as a real human and chat with me, who is a person needs emotional support.' \
               '</s>\n<s>Assistant: '

    def __call__(self, 
        dialogue: str,
        max_new_token: int = 128,
        temperature: int = 0.3
        ) -> str:
        
        input_ids = self.tokenizer([dialogue], return_tensors="pt", add_special_tokens=False).input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        # set the parameters of the generate_input
        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_token,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": temperature,
            "repetition_penalty": 1.3,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }

        generate_ids = self.model.generate(**generate_input)
        text = self.tokenizer.decode(generate_ids[0])
        output = text.split("Assistant: ")[-1].replace("<s>", "").replace("</s>", "").strip()
        
        # avoid the output containing extra 'Human' in one round
        if 'Human' in output:
            output = output.split('Human')[0]
        output = output.rsplit('.', 1)[0] + '.'
        return output