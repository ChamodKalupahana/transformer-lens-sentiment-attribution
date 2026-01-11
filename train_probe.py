import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SAMPLE_LIMIT = 2000

def load_sentiment_data(csv_path, limit=None):
    """
    Loads sentiment data from CSV.
    Returns:
        prompts: list[str]
        labels: torch.Tensor
    """
    df = pd.read_csv(csv_path, encoding='latin-1')
    
    # Filter for positive/negative only
    df = df[df['sentiment'].isin(['positive', 'negative'])]
    
    # Drop NaNs in selected_text (or text)
    col_name = 'selected_text' if 'selected_text' in df.columns else 'text'
    df = df.dropna(subset=[col_name])
    
    # Optional limit
    if limit is not None:
        df = df.head(limit)
    
    # Map sentiment to 0/1 (negative=0, positive=1)
    label_map = {'negative': 0, 'positive': 1}
    labels = df['sentiment'].map(label_map).values
    labels = torch.tensor(labels, dtype=torch.long)
    
    prompts = df[col_name].tolist()
    
    return prompts, labels

@torch.no_grad()
def get_activations(model, prompts, layer: int, batch_size=32):
    """
    Extracts activations from the residual stream at the final token.
    """
    all_acts = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        toks = model.to_tokens(batch_prompts).to(device)
        hook_name = f"blocks.{layer}.hook_resid_post"
        
        logits, cache = model.run_with_cache(toks, names_filter=[hook_name])
        
        acts = cache[hook_name]              # [batch, seq, d_model]
        
        # Identify the last non-padding token position
        if model.tokenizer.pad_token_id is not None:
            pad_id = model.tokenizer.pad_token_id
            mask = (toks != pad_id)
            lengths = mask.sum(dim=1) - 1
        else:
            lengths = torch.full((toks.shape[0],), toks.shape[1] - 1, device=device)

        # Gather activations
        X = acts[torch.arange(acts.shape[0], device=device), lengths, :].detach().cpu().float()
        all_acts.append(X)
        
    return torch.cat(all_acts, dim=0)

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
    # Data
    print("Loading data...")
    train_prompts, train_labels = load_sentiment_data("data/train.csv", limit=TRAIN_SAMPLE_LIMIT)
    val_prompts, val_labels = load_sentiment_data("data/test.csv")
    
    print(f"Train size: {len(train_prompts)}")
    print(f"Val size: {len(val_prompts)}")
    
    layer_accuracies = []
    probes = []
    
    print(f"Looping through {model.cfg.n_layers} layers...")
    for layer in tqdm(range(model.cfg.n_layers)):
        # Extract features
        X_train = get_activations(model, train_prompts, layer, batch_size=128)
        X_val = get_activations(model, val_prompts, layer, batch_size=128)
        
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