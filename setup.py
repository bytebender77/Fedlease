from setuptools import setup, find_packages

setup(
    name="fedlease-finbert",
    version="1.0.0",
    description=(
        "FedLEASE: Federated Adaptive LoRA Expert Allocation and Selection "
        "for Financial NLP (FinBERT + LoRA-MoE)"
    ),
    packages=find_packages(exclude=["experiments", "outputs", "tests"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.1",
        "datasets>=2.15.0,<3.0.0",
        "huggingface_hub>=0.20.0",
        "pyarrow>=14.0.0",
        "scikit-learn>=1.3.2",
        "scipy>=1.11.4",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tqdm>=4.66.0",
        "omegaconf>=2.3.0",
        "PyYAML>=6.0.1",
        "pandas>=2.1.0",
        "accelerate>=0.25.0",
    ],
    entry_points={
        "console_scripts": [
            "fedlease=experiments.run_fedlease:main",
        ],
    },
    extras_require={
        "dev": ["pytest", "black", "isort", "mypy"],
        "tracking": ["wandb>=0.16.0", "tensorboard>=2.15.0"],
    },
)
