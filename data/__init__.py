__all__ = [
    "FinancialDatasetLoader",
    "FinancialPreprocessor",
    "FederatedDataPartitioner",
]


def __getattr__(name):
    if name == "FinancialDatasetLoader":
        from .dataset_loader import FinancialDatasetLoader
        return FinancialDatasetLoader
    if name == "FinancialPreprocessor":
        from .preprocessing import FinancialPreprocessor
        return FinancialPreprocessor
    if name == "FederatedDataPartitioner":
        from .data_partitioner import FederatedDataPartitioner
        return FederatedDataPartitioner
    raise AttributeError(f"module 'data' has no attribute '{name}'")
