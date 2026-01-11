import torch
from transformer_lens import HookedTransformer
from train_probe import get_activations, load_sentiment_data, device
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
import pandas as pd

TEST_SAMPLE_LIMIT = 500

def main():
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    print("Model loaded.")

    print("Loading probes from sentiment_probes.pt...")
    try:
        # Load with weights_only=False as we reuse the fix from model-eval.py
        probes = torch.load("sentiment_probes.pt", map_location=device, weights_only=False)
        print(f"Loaded {len(probes)} probes.")
    except FileNotFoundError:
        print("Error: sentiment_probes.pt not found. Please run train_probe.py first.")
        sys.exit(1)

    print(f"Loading test data (limit={TEST_SAMPLE_LIMIT})...")
    prompts, labels = load_sentiment_data("data/test.csv", limit=TEST_SAMPLE_LIMIT)
    print(f"Test data size: {len(prompts)}")

    print("\nEvaluating probes on test set...")
    print("="*60)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for layer in range(model.cfg.n_layers):
        print(f"Processing Layer {layer}...", end="\r")
        
        # Get activations
        X = get_activations(model, prompts, layer, batch_size=32)
        y_true = labels.cpu().numpy()
        
        # Get predictions
        probe = probes[layer]
        probe.eval()
        probe.to(device) # Ensure probe is on device
        
        with torch.no_grad():
            X = X.to(device)
            logits = probe(X)
            preds = logits.argmax(dim=-1).cpu().numpy()
        
        # Metrics
        acc = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        
        print(f"Layer {layer}: Accuracy = {acc:.3f}")
        print(f"Confusion Matrix (Neg=0, Pos=1):")
        print(cm)
        print("-" * 30)
        
        # Plot
        ax = axes[layer]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax.set_title(f'Layer {layer} (Acc: {acc:.2f})')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('confusion_matrices_subplots.png')
    print("\nSaved confusion matrices to confusion_matrices_subplots.png")

if __name__ == "__main__":
    main()
