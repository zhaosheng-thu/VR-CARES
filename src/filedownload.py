import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download, snapshot_download, login
login_status = login(token="**") # your hf token

def download_file(local_dir, cache_dir, repo_id, filename, repo_type):

    # Download the specific file from the Hugging Face Model Hub
    hf_hub_download(
        cache_dir=cache_dir,
        local_dir=local_dir,
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir_use_symlinks=False,
        resume_download=True,
        )
    
def download_folder(local_dir, cache_dir, repo_id):
    
    snapshot_download(
        cache_dir=cache_dir,
        local_dir=local_dir,
        repo_id=repo_id,
        # local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.model", "*.json", "*.bin",
        "*.py", "*.md", "*.txt", "*.safetensors", "*.msgpack",
        "*.h5", "*.ot", "*.pt", "*.pth", "*.ckpt", "*.yaml"],
        ignore_patterns=[],
        )

def main():
    
    # Specify the model repository and filename
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    cache_dir = "/.cache"
    local_dir = "litgpt/litgpt/model-weights/checkpoints"
    repo_type = None
    
    # download_file(local_dir, cache_dir, repo_id, filename, repo_type)
    download_folder(local_dir, cache_dir, repo_id)


if __name__ == "__main__":
    main()