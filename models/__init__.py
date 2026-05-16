# Lazy imports — the LoRA layers and router are pure-PyTorch and always available.
# The FinBERT models require transformers and are loaded on demand.
__all__ = [
    "LoRAMoELinear",
    "AdaptiveTopMRouter",
    "FedLEASEFinBERT",
    "WarmupFinBERT",
]


def __getattr__(name):
    if name == "LoRAMoELinear":
        from .lora_layers import LoRAMoELinear
        return LoRAMoELinear
    if name == "AdaptiveTopMRouter":
        from .adaptive_router import AdaptiveTopMRouter
        return AdaptiveTopMRouter
    if name in ("FedLEASEFinBERT", "WarmupFinBERT"):
        from .finbert_lora_moe import FedLEASEFinBERT, WarmupFinBERT
        return locals()[name]
    raise AttributeError(f"module 'models' has no attribute '{name}'")
