"""
Run full evaluation: extract embeddings, compute all metrics, generate t-SNE plot.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.model import load_model
from src.dataset import create_dataloaders
from src.evaluate import evaluate_model


def main():
    print("=" * 60)
    print("  AuthNet Full Evaluation")
    print("=" * 60)
    
    # Load best model
    model = load_model(config.BEST_MODEL_PATH, config.DEVICE)
    
    # Create test dataloader
    _, test_loader, _, test_dataset = create_dataloaders()
    
    # Get class names for the t-SNE legend
    class_names = test_dataset.class_names if hasattr(test_dataset, 'class_names') else None
    
    # Run full evaluation
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=config.DEVICE,
        class_names=class_names,
        save_visualizations=True,
    )
    
    print("\nDone! Check outputs/visualizations/ for the t-SNE plot.")


if __name__ == "__main__":
    main()
