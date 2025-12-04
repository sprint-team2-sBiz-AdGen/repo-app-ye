from diffusers import DiffusionPipeline
import torch

model_path = "models_cache/diffusers/photo-background-generation"
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    custom_pipeline=model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
)

print("dir(pipe):", [x for x in dir(pipe) if "token" in x])
print("tokenizer attr:", type(getattr(pipe, "tokenizer", None)))
print("tokenizers attr:", getattr(pipe, "tokenizers", None))