import transformer_lens as tfl
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch as t
device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)


# config = Config
# model-name = "gpt2-small"


cfg = HookedTransformerConfig(
    n_layers=8,
    d_model=512,
    d_head=64,
    n_heads=8,
    d_mlp=2048,
    d_vocab=61,
    n_ctx=59,
    act_fn="gelu",
    normalization_type="LNPre",
    device=device,
)

gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
print(gpt2_small)