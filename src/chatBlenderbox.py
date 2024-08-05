import torch
from transformers import AutoTokenizer
from transformers.models.blenderbot import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import warnings
# 关闭所有警告
warnings.filterwarnings("ignore")

def _norm(x):
    return ' '.join(x.strip().split())

def generate_response(utterances, model, tokenizer, device):

    max_len_tokens = 128
    # Join all the tokens
    input_sequence = ' '.join(
        [' ' + e for e in utterances]) + tokenizer.eos_token  # add space prefix and separate utterances with two spaces
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_sequence))[-max_len_tokens:]
    input_ids = torch.LongTensor([input_ids]).to(device)  # 将输入张量移动到 GPU 上

    model_output = model.generate(input_ids, num_beams=1, do_sample=True, top_p=0.9, num_return_sequences=1,
                                  return_dict=False)
    generation = tokenizer.batch_decode(model_output, skip_special_tokens=True)
    generation = [_norm(e) for e in generation]
    return generation


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BlenderbotTokenizer.from_pretrained('/root/szhao/model-weights/blenderbot-1B-augesc')
    model = BlenderbotForConditionalGeneration.from_pretrained('/root/szhao/model-weights/blenderbot-1B-augesc').to(device)
    model.eval()

    utterances = ["""The following is a conversation with an AI assistant. 
                    The assistant is helpful, empathetic, clever, and very 
                    friendly. It can use various support skills to provide 
                    emotional support to humans"""
    ]

    num_loop = 8
    for turn in range(num_loop):
        print("Turn", turn + 1)
        user_input = input("You: ")
        utterances.append(user_input)
        response = generate_response(utterances=utterances, model=model, tokenizer=tokenizer, device=device)
        choosed_response = response[0] #max(response, key=len)
        print("Bot:", choosed_response)  # each answer
        utterances.append(choosed_response)


if __name__ == "__main__":
    
    main()