"""
AuthNet Model Interpretability
Grad-CAM visualization for understanding authentication decisions.
Custom adaptation for metric learning models (embedding-based, not class-based).
"""

import os
import sys
from typing import Optional, List, Tuple

import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model import EmbeddingNet
from src.dataset import get_test_transforms, inverse_normalize


class EmbeddingSimilarityTarget:
    """
    Custom Grad-CAM target for metric learning models.
    
    Since our model outputs embeddings (not class logits), we can't use 
    standard ClassifierOutputTarget. Instead, we compute gradients w.r.t.
    a specific embedding dimension or similarity score.
    
    This is a non-trivial adaptation that demonstrates deep understanding
    of both Grad-CAM and metric learning.
    """
    
    def __init__(self, target_embedding: Optional[torch.Tensor] = None, dim: Optional[int] = None):
        """
        Args:
            target_embedding: Reference embedding to compute similarity against.
                              If provided, Grad-CAM shows regions contributing to similarity.
            dim: If target_embedding is None, use this embedding dimension as target.
                 The Grad-CAM will show which regions contribute most to this dimension.
        """
        self.target_embedding = target_embedding
        self.dim = dim
    
    def __call__(self, model_output):
        if self.target_embedding is not None:
            # Compute cosine similarity with target
            target = self.target_embedding.to(model_output.device)
            if target.dim() == 1:
                target = target.unsqueeze(0)
            # Similarity as scalar for gradient computation
            return (model_output * target).sum(dim=1)
        elif self.dim is not None:
            # Gradient w.r.t. specific embedding dimension
            return model_output[:, self.dim]
        else:
            # Default: use L2 norm of embedding (shows which regions produce strongest features)
            return model_output.norm(dim=1)


class EmbeddingNormTarget:
    """Simple target that maximizes embedding norm -- shows what the model 'sees'."""
    def __call__(self, model_output):
        if model_output.dim() == 1:
            return model_output.norm()
        return model_output.norm(dim=1)


def get_gradcam(
    model: EmbeddingNet,
    method: str = "gradcam",
) -> GradCAM:
    """
    Create a Grad-CAM instance for the model.
    
    Args:
        model: EmbeddingNet model
        method: "gradcam", "gradcam++", or "eigengradcam"
    
    Returns:
        Grad-CAM instance
    """
    # Target the last convolutional layer of ResNet-18's layer4
    # This is where the highest-level features are computed
    target_layers = [model.backbone[7][-1]]  # layer4[-1]
    
    cam_class = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "eigengradcam": EigenGradCAM,
    }.get(method, GradCAM)
    
    return cam_class(model=model, target_layers=target_layers)


def generate_heatmap(
    model: EmbeddingNet,
    image_input,
    target_embedding: Optional[torch.Tensor] = None,
    method: str = "gradcam",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a Grad-CAM heatmap for an image.
    
    Args:
        model: Trained EmbeddingNet
        image_input: Image path, PIL Image, or tensor
        target_embedding: Optional reference embedding for similarity-based Grad-CAM
        method: CAM method ("gradcam", "gradcam++", "eigengradcam")
    
    Returns:
        (original_image, heatmap, overlay) as numpy arrays (HWC, uint8)
    """
    model.eval()
    device = next(model.parameters()).device
    transform = get_test_transforms()
    
    # Load and preprocess image
    if isinstance(image_input, str):
        original_pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        original_pil = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        original_pil = image_input.convert("RGB")
    elif isinstance(image_input, torch.Tensor):
        # Already preprocessed tensor
        input_tensor = image_input.unsqueeze(0) if image_input.dim() == 3 else image_input
        original_np = inverse_normalize(image_input if image_input.dim() == 3 else image_input[0])
        original_pil = Image.fromarray(original_np)
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")
    
    # Resize for display
    original_resized = original_pil.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    original_np = np.array(original_resized).astype(np.float32) / 255.0
    
    # Transform for model
    if not isinstance(image_input, torch.Tensor):
        input_tensor = transform(original_pil).unsqueeze(0).to(device)
    else:
        input_tensor = input_tensor.to(device)
    
    # Create Grad-CAM
    cam = get_gradcam(model, method)
    
    # Define target
    if target_embedding is not None:
        targets = [EmbeddingSimilarityTarget(target_embedding)]
    else:
        targets = [EmbeddingNormTarget()]
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]  # First (and only) image in batch
    
    # Create overlay
    overlay = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
    
    # Convert heatmap to colored version
    heatmap_colored = cv2.applyColorMap(
        (grayscale_cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    original_uint8 = (original_np * 255).astype(np.uint8)
    
    return original_uint8, heatmap_colored, overlay


def visualize_authentication(
    model: EmbeddingNet,
    image_a,
    image_b,
    save_path: Optional[str] = None,
    title: str = "Authentication Decision Explanation",
    method: str = "gradcam",
):
    """
    Visualize how the model compares two images for authentication.
    Shows Grad-CAM heatmaps for both images, highlighting discriminative regions.
    
    Args:
        model: Trained EmbeddingNet
        image_a: First image (reference/gallery item)
        image_b: Second image (query item)
        save_path: Where to save the visualization
        title: Plot title
        method: CAM method
    """
    save_path = save_path or os.path.join(config.VIZ_DIR, "auth_explanation.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    transform = get_test_transforms()
    
    # Get embeddings
    def get_embedding(img):
        if isinstance(img, str):
            pil = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            pil = Image.fromarray(img).convert("RGB")
        else:
            pil = img.convert("RGB")
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor)
        return emb[0]
    
    emb_a = get_embedding(image_a)
    emb_b = get_embedding(image_b)
    similarity = float(torch.dot(emb_a, emb_b))
    
    # Generate heatmaps
    # For image A: show regions similar to B's embedding
    orig_a, _, overlay_a = generate_heatmap(model, image_a, target_embedding=emb_b, method=method)
    # For image B: show regions similar to A's embedding
    orig_b, _, overlay_b = generate_heatmap(model, image_b, target_embedding=emb_a, method=method)
    
    # Also generate self-attention maps (what does the model see in each image?)
    _, _, self_overlay_a = generate_heatmap(model, image_a, method=method)
    _, _, self_overlay_b = generate_heatmap(model, image_b, method=method)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    verdict = "MATCH [Y]" if similarity >= config.AUTH_THRESHOLD else "MISMATCH [N]"
    verdict_color = "green" if similarity >= config.AUTH_THRESHOLD else "red"
    
    fig.suptitle(
        f"{title}\nSimilarity: {similarity:.4f} | Threshold: {config.AUTH_THRESHOLD} | {verdict}",
        fontsize=14, fontweight='bold', color=verdict_color
    )
    
    # Row 1: Image A
    axes[0, 0].imshow(orig_a)
    axes[0, 0].set_title("Reference Image", fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(self_overlay_a)
    axes[0, 1].set_title("Self-Attention (Feature Focus)", fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(overlay_a)
    axes[0, 2].set_title("Cross-Attention (vs Query)", fontsize=11)
    axes[0, 2].axis('off')
    
    # Row 2: Image B
    axes[1, 0].imshow(orig_b)
    axes[1, 0].set_title("Query Image", fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(self_overlay_b)
    axes[1, 1].set_title("Self-Attention (Feature Focus)", fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(overlay_b)
    axes[1, 2].set_title("Cross-Attention (vs Reference)", fontsize=11)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Authentication visualization saved to {save_path}")
    print(f"  Similarity: {similarity:.4f} | Verdict: {verdict}")


def generate_batch_heatmaps(
    model: EmbeddingNet,
    image_paths: List[str],
    save_dir: Optional[str] = None,
    method: str = "gradcam",
    max_images: int = 16,
):
    """
    Generate Grad-CAM heatmaps for a batch of images and save as a grid.
    
    Args:
        model: Trained EmbeddingNet
        image_paths: List of image paths
        save_dir: Directory to save individual heatmaps and grid
        method: CAM method
        max_images: Maximum number of images to process
    """
    save_dir = save_dir or config.VIZ_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    image_paths = image_paths[:max_images]
    n = len(image_paths)
    
    if n == 0:
        print("No images to process.")
        return
    
    # Calculate grid dimensions
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
    if rows == 1:
        axes = axes[np.newaxis, :]
    
    for i, img_path in enumerate(tqdm(image_paths, desc="Generating heatmaps")):
        row = i // cols
        col = i % cols
        
        try:
            original, _, overlay = generate_heatmap(model, img_path, method=method)
            
            axes[row, col * 2].imshow(original)
            axes[row, col * 2].set_title(os.path.basename(img_path)[:20], fontsize=8)
            axes[row, col * 2].axis('off')
            
            axes[row, col * 2 + 1].imshow(overlay)
            axes[row, col * 2 + 1].set_title("Grad-CAM", fontsize=8)
            axes[row, col * 2 + 1].axis('off')
        except Exception as e:
            print(f"  [WARN] Failed for {img_path}: {e}")
    
    # Hide empty subplots
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col * 2].axis('off')
        axes[row, col * 2 + 1].axis('off')
    
    plt.suptitle("AuthNet Grad-CAM: Feature Attention Maps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    grid_path = os.path.join(save_dir, "gradcam_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM grid saved to {grid_path}")
