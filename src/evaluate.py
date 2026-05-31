"""
AuthNet Evaluation Module
Computes retrieval metrics: Recall@K, mAP, NMI.
Generates t-SNE visualization of the embedding space.
"""

import os
import sys
import json
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = None,
) -> Dict[int, float]:
    """
    Compute Recall@K retrieval metric.
    
    For each query, find the K nearest neighbors in embedding space.
    Recall@K = fraction of queries with at least one correct match in top-K.
    
    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of class labels
        k_values: List of K values to compute (default: [1, 2, 4, 8])
    
    Returns:
        Dict mapping K → Recall@K value
    """
    k_values = k_values or config.RECALL_K
    n_samples = len(embeddings)
    
    if n_samples == 0:
        return {k: 0.0 for k in k_values}
    
    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Set self-similarity to -inf to exclude self-matches
    np.fill_diagonal(sim_matrix, -np.inf)
    
    # Get indices sorted by similarity (descending)
    sorted_indices = np.argsort(-sim_matrix, axis=1)
    
    recall_dict = {}
    
    for k in k_values:
        if k >= n_samples:
            recall_dict[k] = 1.0
            continue
        
        # For each query, check if any of top-K neighbors has the same label
        correct = 0
        for i in range(n_samples):
            top_k_indices = sorted_indices[i, :k]
            top_k_labels = labels[top_k_indices]
            
            if labels[i] in top_k_labels:
                correct += 1
        
        recall_dict[k] = correct / n_samples
    
    return recall_dict


def compute_map(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Mean Average Precision (mAP) for retrieval.
    
    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of class labels
    
    Returns:
        mAP score
    """
    n_samples = len(embeddings)
    if n_samples == 0:
        return 0.0
    
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)
    sorted_indices = np.argsort(-sim_matrix, axis=1)
    
    average_precisions = []
    
    for i in range(n_samples):
        query_label = labels[i]
        sorted_labels = labels[sorted_indices[i]]
        
        # Compute precision at each relevant position
        relevant = (sorted_labels == query_label)
        n_relevant = relevant.sum()
        
        if n_relevant == 0:
            continue
        
        cumsum = np.cumsum(relevant)
        precision_at_k = cumsum / np.arange(1, len(sorted_labels) + 1)
        
        # Average precision = mean of precisions at relevant positions
        ap = np.sum(precision_at_k * relevant) / n_relevant
        average_precisions.append(ap)
    
    return np.mean(average_precisions) if average_precisions else 0.0


def compute_nmi(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Normalized Mutual Information (NMI) via K-Means clustering.
    
    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of ground truth labels
    
    Returns:
        NMI score
    """
    n_clusters = len(np.unique(labels))
    
    if n_clusters <= 1:
        return 0.0
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted = kmeans.fit_predict(embeddings)
    
    return normalized_mutual_info_score(labels, predicted)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "t-SNE Embedding Space Visualization",
    max_classes: int = 30,
    perplexity: int = 30,
):
    """
    Generate and save a t-SNE visualization of the embedding space.
    
    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of class labels
        class_names: Optional list mapping label index → class name
        save_path: Where to save the plot
        title: Plot title
        max_classes: Max number of classes to show (for readability)
        perplexity: t-SNE perplexity parameter
    """
    save_path = save_path or os.path.join(config.VIZ_DIR, "tsne_embeddings.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Limit classes for readability
    unique_labels = np.unique(labels)
    if len(unique_labels) > max_classes:
        selected_labels = np.random.choice(unique_labels, max_classes, replace=False)
        mask = np.isin(labels, selected_labels)
        embeddings = embeddings[mask]
        labels = labels[mask]
        unique_labels = selected_labels
    
    # Adjust perplexity if needed
    n_samples = len(embeddings)
    perplexity = min(perplexity, max(5, n_samples // 4))
    
    print(f"Computing t-SNE (n={n_samples}, perplexity={perplexity})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use a good colormap
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        # Shorten long names
        if len(name) > 20:
            name = name[:17] + "..."
        
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[cmap(idx)],
            label=name,
            alpha=0.7,
            s=20,
            edgecolors='none',
        )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    
    # Legend (only if reasonable number of classes)
    if len(unique_labels) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, 
                  markerscale=2, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE plot saved to {save_path}")


@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
    device: torch.device = None,
    class_names: Optional[List[str]] = None,
    save_visualizations: bool = True,
) -> dict:
    """
    Full evaluation pipeline: extract embeddings → compute all metrics → visualize.
    
    Returns:
        dict with all evaluation metrics
    """
    device = device or config.DEVICE
    model.eval()
    model = model.to(device)
    
    # Extract embeddings
    print("\nExtracting embeddings from test set...")
    all_embeddings = []
    all_labels = []
    
    for images, labels in tqdm(test_loader, desc="Extracting", ncols=100):
        images = images.to(device)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
    
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    print(f"Extracted {len(all_embeddings)} embeddings of dim {all_embeddings.shape[1]}")
    
    # Compute metrics
    print("\nComputing retrieval metrics...")
    recall_dict = compute_recall_at_k(all_embeddings, all_labels)
    map_score = compute_map(all_embeddings, all_labels)
    nmi_score = compute_nmi(all_embeddings, all_labels)
    
    results = {
        'recall': recall_dict,
        'mAP': map_score,
        'NMI': nmi_score,
        'n_samples': len(all_embeddings),
        'n_classes': len(np.unique(all_labels)),
        'embedding_dim': all_embeddings.shape[1],
    }
    
    # Print results
    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    for k, v in recall_dict.items():
        print(f"  Recall@{k}: {v:.4f} ({v:.2%})")
    print(f"  mAP:       {map_score:.4f} ({map_score:.2%})")
    print(f"  NMI:       {nmi_score:.4f}")
    print(f"  Samples:   {results['n_samples']}")
    print(f"  Classes:   {results['n_classes']}")
    print("=" * 50)
    
    # Save results
    results_path = os.path.join(config.OUTPUT_DIR, "evaluation_results.json")
    
    # Convert keys to strings for JSON serialization
    json_results = {
        'recall': {str(k): float(v) for k, v in recall_dict.items()},
        'mAP': float(map_score),
        'NMI': float(nmi_score),
        'n_samples': int(results['n_samples']),
        'n_classes': int(results['n_classes']),
    }
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Visualizations
    if save_visualizations:
        plot_tsne(
            all_embeddings, all_labels, 
            class_names=class_names,
            title="AuthNet: Texture Embedding Space (t-SNE)",
        )
    
    return results
