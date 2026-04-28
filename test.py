from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

model_id = "ModelOrganismsForEM/Qwen2.5-32B-Instruct_bad-medical-advice"
SAVE_PATH = Path("/media/am/AM/models/EM_bad_FULL")

# load directly (already has LoRA applied internally)
"""model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

# 🔥 CRITICAL: disable PEFT saving logic
model._hf_peft_config_loaded = False

# save
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
"""
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(SAVE_PATH, device_map="cpu", local_files_only=True)
inputs = tokenizer("Hello", return_tensors="pt")
out = model(**inputs, output_hidden_states=True)

print(out.logits.shape)
print(len(out.logits))
for k in out.logits:
    print(k)