from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

model_path = Path(__file__).resolve().parents[2] / "models" / "t5_small"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

model.eval()


def predict(input_text: str) -> str:
    # Tokenize the input text
    prompt = "summarize: " + input_text
    encoded = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_length=128,
            num_beams=4,
            length_penalty=0.6,
            early_stopping=True,
        )

    # Decode the generated tokens to text
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_text