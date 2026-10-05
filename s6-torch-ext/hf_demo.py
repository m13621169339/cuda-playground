import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name, torch_dtype=torch.float16, device_map="cuda")

print(f"权重加载后显存: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e9:.2f} B")

prompt = "用一句话解释GPU为什么适合训练大模型"
inputs = tok(prompt, return_tensors="pt").to("cuda")
print(f"输入 token 数: {inputs['input_ids'].shape[1]}")

torch.cuda.synchronize(); t0 = time.perf_counter()
out = model.generate(**inputs, max_new_tokens=100)
torch.cuda.synchronize(); t1 = time.perf_counter()

print("=" * 40)
print(tok.decode(out[0], skip_special_tokens=True))
print("=" * 40)
print(f"生成 100 token 耗时: {t1-t0:.2f} s ({100/(t1-t0):.0f} token/s)")
print(f"峰值显存: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
