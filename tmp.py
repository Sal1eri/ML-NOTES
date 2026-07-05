import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

text = "hello world"

inputs = tokenizer(text, return_tensors="pt").to(model.device)

input_ids = inputs["input_ids"][0]

print("TOKENS + IDS:")
tokens = tokenizer.convert_ids_to_tokens(input_ids)

for tok, tid in zip(tokens, input_ids.tolist()):
    print(f"{tok:15s} -> {tid}")

print("\n" + "="*60)

# embedding matrix
embed = model.model.embed_tokens.weight  # [vocab, hidden]

print("Embedding matrix shape:", embed.shape)

print("\nTOKEN -> EMBEDDING (raw numbers)\n")

for tok, tid in zip(tokens, input_ids.tolist()):
    vec = embed[tid]  # tensor [hidden]

    print(f"\nTOKEN: {tok} | ID: {tid}")
    print("FIRST 20 DIMENSIONS:")
    print(vec[:20].detach().float().cpu().numpy())