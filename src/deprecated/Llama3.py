import transformers
import torch

model_id = "/root/szhao/model-weights/Llama-3-8B"
device_map = "cuda:0" if torch.cuda.is_available() else "auto"
pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map=device_map
)

pipeline("Hey how are you doing today?")
print(f"\nMemory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")