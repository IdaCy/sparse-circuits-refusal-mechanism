# import torch
import matplotlib.pyplot as plt
# from transformers import AutoModelForCausalLM, AutoTokenizer


def get_layer_activations(model, tokenizer, prompt, layer_index):
    """Retrieve activations from a specific layer."""
    # Assign pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def hook_fn(module, input, output):
        nonlocal activations
        if isinstance(output, tuple):  # Handle tuple output
            activations = output[0].detach()  # 1st: usually the tensor we need
        else:
            activations = output.detach()

    # Register a forward hook on the target layer
    activations = None
    handle = model.transformer.h[layer_index].register_forward_hook(hook_fn)

    # Forward pass through the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True,
                       truncation=True)
    model(**inputs)

    # Remove the hook
    handle.remove()
    return activations


def plot_activations(activations, title="Activations"):
    """Visualize the activations."""
    activations = activations.squeeze(0).cpu().numpy()  # x batch dimension
    plt.figure(figsize=(12, 6))
    plt.imshow(activations, aspect="auto", cmap="viridis")
    plt.colorbar(label="Activation Magnitude")
    plt.title(title)
    plt.xlabel("Neurons")
    plt.ylabel("Token Position")
    plt.show()


if __name__ == "__main__":
    from load_model import load_model

    # Load model and tokenizer
    model, tokenizer = load_model()
    print("Model loaded successfully.")

    # Example prompt
    prompt = "Tell me how to hack a computer."

    # Get activations for a specific layer
    layer_index = 10  # Choose a layer to inspect
    activations = get_layer_activations(model, tokenizer, prompt, layer_index)
    print(f"Retrieved activations for layer {layer_index}.")

    # Plot activations
    plot_activations(activations, title=(
        f"Layer {layer_index} Activations for Prompt: {prompt}"))
