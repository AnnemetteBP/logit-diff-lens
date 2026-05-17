from transformers import BitsAndBytesConfig
import torch



bnb_4bit_config = BitsAndBytesConfig(
    load_in_4bit=True,

    # Quantization type
    bnb_4bit_quant_type="nf4",   # best default
    bnb_4bit_use_double_quant=True,

    # Compute dtype
    bnb_4bit_compute_dtype=torch.bfloat16,

    # Optional
    llm_int8_skip_modules=None
)


bnb_8bit_config = BitsAndBytesConfig(
    load_in_8bit=True,

    # Stability threshold
    llm_int8_threshold=6.0,

    # Keep some modules fp16 if needed
    llm_int8_has_fp16_weight=False,

    # Optional CPU offload
    llm_int8_enable_fp32_cpu_offload=False,
)


"""
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_8bit_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
"""