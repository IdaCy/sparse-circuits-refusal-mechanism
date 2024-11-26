#  import torch

def analyze_residuals(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    print(f"Number of layers: {len(hidden_states)}")
    return hidden_states


if __name__ == "__main__":
    from load_model import load_model
    model, tokenizer = load_model()
    hidden_states = analyze_residuals(model, tokenizer, "Example prompt")
