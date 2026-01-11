import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_synthetic_sentiment_data(n_samples=500):
    """
    Generates simple template-based sentiment data.
    Returns:
        prompts: list[str]
        labels: torch.Tensor
    """
    positive_adjs = ["good", "great", "amazing", "lovely", "wonderful", "fantastic", "superb", "excellent", "brilliant", "perfect"]
    negative_adjs = ["bad", "terrible", "awful", "horrible", "disgusting", "dreadful", "poor", "abysmal", "lousy", "useless"]
    
    subjects = ["movie", "book", "food", "place", "service", "game", "show", "performance", "idea", "product"]
    verbs = ["is", "was", "looks", "seems", "feels", "smells", "tastes"]
    
    prompts = []
    labels = []
    
    for _ in range(n_samples):
        label = random.randint(0, 1) # 1 = Positive, 0 = Negative
        adj = random.choice(positive_adjs) if label == 1 else random.choice(negative_adjs)
        subj = random.choice(subjects)
        verb = random.choice(verbs)
        
        # Randomize template
        template_id = random.randint(0, 2)
        if template_id == 0:
            text = f"The {subj} {verb} {adj}."
        elif template_id == 1:
            text = f"I think this {subj} is {adj}."
        else:
            text = f"That was a {adj} {subj}."
            
        prompts.append(text)
        labels.append(label)
        
    return prompts, torch.tensor(labels)

@torch.no_grad()
def get_activations(model, prompts, layer: int):
    """
    Extracts activations from the residual stream at the final token.
    """
    toks = model.to_tokens(prompts).to(device)
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    logits, cache = model.run_with_cache(toks, names_filter=[hook_name])
    
    acts = cache[hook_name]              # [batch, seq, d_model]
    # Use the activation of the last token (which predicts the next token, or contains sentence context)
    # Note: For classification sometimes we use the last token.
    X = acts[:, -1, :].detach().cpu().float()
    return X

def train_linear_probe(X_train, y_train, X_val, y_val, epochs=50, lr=1e-3, batch_size=32, verbose=False):
    n_classes = int(y_train.max().item()) + 1
    probe = nn.Linear(X_train.shape[1], n_classes).to(device)
    
    # Move data to device used for training probe (can be CPU if small, but let's use device)
    # Actually for small linear probes CPU is often fast enough and avoids transferring huge activation dumps if they were big.
    # But let's put them on device for consistency if they fit.
    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)
    X_val_d = X_val.to(device)
    y_val_d = y_val.to(device)

    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train_d, y_train_d)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        probe.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = probe(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    
    # Eval
    probe.eval()
    with torch.no_grad():
        logits_val = probe(X_val_d)
        preds = logits_val.argmax(dim=-1)
        acc = (preds == y_val_d).float().mean().item()
        
    return acc

def main():
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    print("Model loaded.")

    # Data
    print("Generating data...")
    prompts, labels = get_synthetic_sentiment_data(n_samples=500)
    
    # Split
    n = len(prompts)
    indices = torch.randperm(n)
    train_size = int(0.8 * n)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    train_prompts = [prompts[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_prompts = [prompts[i] for i in val_idx]
    val_labels = labels[val_idx]
    
    layer_accuracies = []
    
    print(f"Looping through {model.cfg.n_layers} layers...")
    for layer in tqdm(range(model.cfg.n_layers)):
        # Extract features
        X_train = get_activations(model, train_prompts, layer)
        X_val = get_activations(model, val_prompts, layer)
        
        # Train probe
        acc = train_linear_probe(X_train, train_labels, X_val, val_labels, epochs=30, lr=1e-3)
        layer_accuracies.append(acc)
        # print(f"Layer {layer}: Acc {acc:.3f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(model.cfg.n_layers), layer_accuracies, marker='o')
    plt.title("Probe Accuracy by Layer (Sentiment)")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.savefig("sentiment_probe_results.png")
    print("Results saved to sentiment_probe_results.png")
    
    # Print table
    print("\nLayer | Accuracy")
    print("------|---------")
    for i, acc in enumerate(layer_accuracies):
        print(f"{i:5d} | {acc:.3f}")

if __name__ == "__main__":
    main()