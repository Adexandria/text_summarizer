from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.float16
)
model = PeftModel.from_pretrained(
    base_model,
    Path(__file__).resolve().parents[2] / "models" / "qwen",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

SYSTEM_PROMPT = (
    "You are a concise text summarizer. Read the step-by-step instructions "
    "and output ONLY a short, imperative text that summarizes."
)


def predict(text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Instructions:\n{text}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary