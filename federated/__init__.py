# Lazy imports — avoids pulling in torch/transformers at module discovery time
__all__ = [
    "FederatedClient",
    "FederatedServer",
    "aggregate_expert_states",
    "aggregate_router_states",
]


def __getattr__(name):
    if name == "FederatedClient":
        from .client import FederatedClient
        return FederatedClient
    if name == "FederatedServer":
        from .server import FederatedServer
        return FederatedServer
    if name in ("aggregate_expert_states", "aggregate_router_states"):
        from .aggregation import aggregate_expert_states, aggregate_router_states
        return locals()[name]
    raise AttributeError(f"module 'federated' has no attribute '{name}'")
