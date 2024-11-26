from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model()
    print(f"Loaded model: {model.config.name_or_path}")
