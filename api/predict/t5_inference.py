from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

def load_model(model_name: str = "t5_small"):
    model_path = Path(__file__).resolve().parents[2] / "models" / model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    
    return model, tokenizer, device



def predict(model_name: str ,input_text: str) -> str:
    model, tokenizer, device = load_model(model_name)
    
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