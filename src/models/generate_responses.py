import json
# from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(filename):
    """Load prompts from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def generate_response(model, tokenizer, prompt, max_length=50):
    """Generate a response for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def save_responses(responses, filename):
    """Save generated responses to a JSON file."""
    with open(filename, "w") as f:
        json.dump(responses, f, indent=4)
    print(f"Saved responses to {filename}")


if __name__ == "__main__":
    from load_model import load_model

    # Load model and tokenizer
    model, tokenizer = load_model()
    print("Model loaded successfully.")

    # Load prompts
    prompts = load_prompts("src/data/prompts.json")
    print(f"Loaded {len(prompts)} prompts.")

    # Generate responses
    responses = []
    for item in prompts:
        prompt = item["prompt"]
        response = generate_response(model, tokenizer, prompt)
        responses.append({"prompt": prompt, "response": response,
                          "label": item["label"]})
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")

    # Save responses
    save_responses(responses, "src/data/responses.json")
