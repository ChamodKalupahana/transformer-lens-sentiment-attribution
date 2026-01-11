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
    
    # Identify the last non-padding token position
    # GPT-2 tokenizer usually uses EOS as BOS/PAD, but let's check explicitly
    if model.tokenizer.pad_token_id is not None:
        pad_id = model.tokenizer.pad_token_id
        # Create mask of non-pad tokens
        mask = (toks != pad_id)
        # Find index of last true
        # We can sum the mask to get length (assuming right padding or no padding in middle)
        # Note: attention_mask usually handles this, but here we work with tokens directly
        # If padded on right: last_idx = sum(mask) - 1
        lengths = mask.sum(dim=1) - 1
    else:
        # If no pad token, assume full length
        lengths = torch.full((toks.shape[0],), toks.shape[1] - 1, device=device)

    # Gather the activations at the correct positions
    # acts[i, lengths[i], :]
    X = acts[torch.arange(acts.shape[0], device=device), lengths, :].detach().cpu().float()
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
        
    return probe, acc

def get_sentiment_score(probe, activation):
    """
    Returns a score between -1 (negative) and 1 (positive).
    """
    probe.eval()
    with torch.no_grad():
        activation = activation.to(device)
        logits = probe(activation)
        probs = torch.softmax(logits, dim=-1)
        # Assuming class 1 is positive and class 0 is negative
        # score = p(1) - p(0)
        # which is equivalent to 2*p(1) - 1
        score = probs[0, 1].item() - probs[0, 0].item()
    return score

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
    probes = []
    
    print(f"Looping through {model.cfg.n_layers} layers...")
    for layer in tqdm(range(model.cfg.n_layers)):
        # Extract features
        X_train = get_activations(model, train_prompts, layer)
        X_val = get_activations(model, val_prompts, layer)
        
        # Train probe
        probe, acc = train_linear_probe(X_train, train_labels, X_val, val_labels, epochs=30, lr=1e-3)
        layer_accuracies.append(acc)
        # print(f"Layer {layer}: Acc {acc:.3f}")
        
        # Save probe (or keep in memory)
        # For simplicity, we'll just keep them in a list if memory allows (small model, linear probes are tiny)
        from types import SimpleNamespace
        probes.append(probe)   
    
    # Store probes in a dictionary or list for easy access
    # probes is already a list indexed by layer

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


    # Save probes
    print("Saving probes to sentiment_probes.pt...")
    torch.save(probes, "sentiment_probes.pt")
    print("Probes saved.")

if __name__ == "__main__":
    main()