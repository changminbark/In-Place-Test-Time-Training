from huggingface_hub import hf_hub_download
from safetensors import safe_open

path = hf_hub_download("changminbark/gemma-3-1b-it-ttt-longalpaca-full", "model.safetensors")
with safe_open(path, framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(key, tensor.shape, tensor.dtype)
        if "ttt" in key:
            print(tensor)