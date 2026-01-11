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

    
        # Intervention Loop
        while True:
            intervene_cmd = input("\nDo you want to intervene? (y/n): ").strip().lower()
            if intervene_cmd != 'y':
                break
                
            try:
                layers_input = input("Enter layers to intervene on (comma separated, e.g. 6,7,8): ")
                target_layers = [int(x.strip()) for x in layers_input.split(',')]
                
                strength = float(input("Enter intervention strength (positive for pos, negative for neg): "))
            except ValueError:
                print("Invalid input. Please try again.")
                continue
                
            print(f"\nGenerations with intervention (Layers: {target_layers}, Strength: {strength}):")
            
            # Define hooks
            hooks = []
            for layer in target_layers:
                if layer < 0 or layer >= model.cfg.n_layers:
                    continue
                    
                # Get direction
                probe = probes[layer]
                # weight shape: [out_features, in_features] -> [2, d_model]
                # Direction: Positive (index 1) - Negative (index 0)
                direction = probe.weight[1] - probe.weight[0] 
                direction = direction.to(device)
                # Normalize direction
                direction = direction / direction.norm()
                
                hook_name = f"blocks.{layer}.hook_resid_post"
                
                def intervention_hook(value, hook, dir_vec=direction, alpha=strength):
                    # value shape: [batch, seq, d_model]
                    # Add direction to all tokens
                    return value + alpha * dir_vec
                    
                hooks.append((hook_name, intervention_hook))
                
            # Generate with hooks
            with model.hooks(fwd_hooks=hooks):
                gen_text_intervened = model.generate(user_input, max_new_tokens=n_tokens, verbose=False)
                
            print(f"Original:  {gen_text}")
            print(f"Intervened: {gen_text_intervened}")
            print("-" * 50)
            
        print("\n") # End of prompt loop iteration

def intervene(layers: [int]):
    """
    Placeholder for potential future use function
    """
    pass


if __name__ == "__main__":
    main()