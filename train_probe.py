import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model.eval()  # frozen model

@torch.no_grad()
def get_X_y(prompts, labels, layer: int, hook_name=None):
    """
    prompts: list[str]
    labels:  torch.Tensor [batch] with class ids (e.g. 0/1)
    returns: X [batch, d_model], y [batch]
    """
    toks = model.to_tokens(prompts).to(device)

    # pick hook point
    hook_name = hook_name or f"blocks.{layer}.hook_resid_post"

    logits, cache = model.run_with_cache(toks, names_filter=[hook_name])

    acts = cache[hook_name]              # [batch, seq, d_model]
    X = acts[:, -1, :].detach().cpu()    # final token position
    y = labels.detach().cpu()
    return X, y

def train_linear_probe(X_train, y_train, X_val, y_val, epochs=20, lr=1e-2, batch_size=256):
    n_classes = int(y_train.max().item()) + 1
    probe = nn.Linear(X_train.shape[1], n_classes)

    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    probe.train()
    for ep in range(epochs):
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = probe(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # quick val accuracy
        probe.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = probe(xb).argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        acc = correct / total
        probe.train()
        print(f"epoch {ep+1:02d} val_acc={acc:.3f}")

    return probe

if __name__ == "__main__":
    # Suppose you have prompts and binary labels already:
    prompts = ["When John and Mary ...", "..."]
    labels  = torch.tensor([1, 0])  # e.g. 1=Mary is correct, 0=John is correct

    # split train/val
    n = len(prompts)
    idx = torch.randperm(n)
    train_idx, val_idx = idx[: int(0.8*n)], idx[int(0.8*n):]

    X_train, y_train = get_X_y([prompts[i] for i in train_idx], labels[train_idx], layer=8)
    X_val, y_val     = get_X_y([prompts[i] for i in val_idx],   labels[val_idx],   layer=8)

    probe = train_linear_probe(X_train, y_train, X_val, y_val)
