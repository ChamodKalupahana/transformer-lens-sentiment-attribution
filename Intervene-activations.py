import torch
from transformer_lens import HookedTransformer
from train_probe import get_activations, get_sentiment_score, device
import sys

def main():
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    print("Model loaded.")

    try:
        # We perform a full load of model objects, so weights_only=False is needed (and safe here)
        probes = torch.load("sentiment_probes.pt", map_location=device, weights_only=False)
        print(f"Loaded {len(probes)} probes.")
    except FileNotFoundError:
        print("Error: sentiment_probes.pt not found. Please run train_probe.py first.")
        sys.exit(1)

    # Interactive Loop
    print("\n" + "="*50)
    print("Interactive Sentiment Analysis")
    print("Enter a sentence to see how the model perceives its sentiment across layers.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("Enter prompt: ")
        except EOFError:
            break
            
        if user_input.lower() in ["exit", "quit", ""]:
            break
        if not user_input.strip():
            continue
            
        print("\nAnalyzing...", end="\r")
        
        scores = []
        for layer in range(model.cfg.n_layers):
            # Get activation for single prompt
            act = get_activations(model, [user_input], layer)
            probe = probes[layer]
            # Ensure probe is on correct device (should be if loaded with map_location, but good to check/move if flexible)
            probe.to(device) 
            score = get_sentiment_score(probe, act)
            scores.append(score)
            
        print(f"Analysis for: '{user_input}'")
        print("Layer | Sentiment Score (-1 to 1) | Interpretation")
        print("------|---------------------------|---------------")
        for i, score in enumerate(scores):
            interpretation = "Positive" if score > 0.5 else "Negative" if score < -0.5 else "Neutral/Mixed"
            bar_len = int((score + 1) * 10) # 0 to 20
            # Clamp bar_len to valid range 0-20
            bar_len = max(0, min(20, bar_len))
            bar = "*" * bar_len + "." * (20 - bar_len)
            print(f"{i:5d} | {score:6.3f} [{bar}] | {interpretation}")
        print("\n")
        
        # Generation
        print("Generating text...", end="\r")
        n_tokens = 20
        gen_text = model.generate(user_input, max_new_tokens=n_tokens, verbose=False)
        print(f"Generated ({n_tokens} tokens):")
        print(f"{gen_text}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()