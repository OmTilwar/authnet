"""
AuthNet End-to-End Demo Script
Demonstrates the complete pipeline: fingerprinting, authentication, and interpretability.
"""

import os
import sys
import random

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model import load_model, build_model
from src.dataset import TextureDataset, get_test_transforms
from src.fingerprint import FingerprintEngine
from src.interpretability import (
    generate_heatmap, visualize_authentication, generate_batch_heatmaps
)
from src.evaluate import evaluate_model, plot_tsne


def run_demo():
    """Run the complete AuthNet demonstration."""
    print("=" * 60)
    print("  AuthNet: Product Authentication & Visual Fingerprinting")
    print("  End-to-End Demo")
    print("=" * 60)
    
    # ── 1. Load Model ──
    print("\n[1/5] Loading model...")
    if os.path.exists(config.BEST_MODEL_PATH):
        model = load_model(config.BEST_MODEL_PATH, config.DEVICE)
        print(f"  Loaded trained model from {config.BEST_MODEL_PATH}")
    else:
        print("  No trained model found. Using pre-trained backbone (demo mode).")
        model = build_model(config.DEVICE)
    
    # ── 2. Initialize Fingerprint Engine ──
    print("\n[2/5] Initializing FingerprintEngine...")
    engine = FingerprintEngine(model=model, device=config.DEVICE)
    
    # ── 3. Find test images ──
    test_dir = os.path.join(config.COMBINED_DIR, "test")
    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
        print("  [WARN] No test data found. Creating synthetic demo images...")
        _create_demo_images()
        test_dir = os.path.join(config.DATA_DIR, "demo")
    
    # Collect image paths by class
    images_by_class = {}
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        imgs = [os.path.join(class_path, f) for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if imgs:
            images_by_class[class_name] = imgs
    
    if not images_by_class:
        print("  [ERROR] No images found for demo.")
        return
    
    print(f"  Found {sum(len(v) for v in images_by_class.values())} images "
          f"across {len(images_by_class)} classes")
    
    # ── 4. Demo: Fingerprinting ──
    print("\n[3/5] Fingerprinting Demo...")
    class_names = list(images_by_class.keys())
    sample_class = random.choice(class_names)
    sample_images = images_by_class[sample_class][:3]
    
    print(f"  Generating fingerprints for class: {sample_class}")
    for img_path in sample_images:
        fp = engine.create_fingerprint(img_path)
        print(f"    {os.path.basename(img_path):>30s} → "
              f"[{fp[0]:.3f}, {fp[1]:.3f}, {fp[2]:.3f}, ... ] "
              f"(dim={len(fp)}, norm={np.linalg.norm(fp):.4f})")
    
    # ── 5. Demo: Authentication ──
    print("\n[4/5] Authentication Demo...")
    
    # Same-class pair (should MATCH)
    if len(sample_images) >= 2:
        result = engine.verify_pair(sample_images[0], sample_images[1])
        print(f"  Same class ({sample_class}):")
        print(f"    Similarity: {result['similarity']:.4f}")
        print(f"    Verdict:    {result['verdict']}")
    
    # Cross-class pair (should MISMATCH)
    if len(class_names) >= 2:
        other_class = [c for c in class_names if c != sample_class][0]
        other_image = images_by_class[other_class][0]
        
        result = engine.verify_pair(sample_images[0], other_image)
        print(f"\n  Cross class ({sample_class} vs {other_class}):")
        print(f"    Similarity: {result['similarity']:.4f}")
        print(f"    Verdict:    {result['verdict']}")
    
    # ── 6. Demo: Grad-CAM Interpretability ──
    print("\n[5/5] Grad-CAM Interpretability Demo...")
    
    # Single image heatmap
    try:
        original, heatmap, overlay = generate_heatmap(model, sample_images[0])
        heatmap_path = os.path.join(config.VIZ_DIR, "demo_heatmap.png")
        Image.fromarray(overlay).save(heatmap_path)
        print(f"  Single heatmap saved to {heatmap_path}")
    except Exception as e:
        print(f"  [WARN] Heatmap generation failed: {e}")
    
    # Authentication visualization (pair comparison)
    if len(sample_images) >= 2:
        try:
            visualize_authentication(
                model, sample_images[0], sample_images[1],
                save_path=os.path.join(config.VIZ_DIR, "demo_auth_comparison.png"),
                title="Demo: Same-Class Authentication",
            )
        except Exception as e:
            print(f"  [WARN] Auth visualization failed: {e}")
    
    # Batch heatmaps
    all_images = []
    for imgs in images_by_class.values():
        all_images.extend(imgs[:2])
    
    if all_images:
        try:
            generate_batch_heatmaps(
                model, all_images[:8],
                save_dir=config.VIZ_DIR,
            )
        except Exception as e:
            print(f"  [WARN] Batch heatmaps failed: {e}")
    
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print(f"  Visualizations saved to: {config.VIZ_DIR}")
    print("  To start the API: uvicorn api.main:app --reload")


def _create_demo_images():
    """Create simple demo images if no dataset is available."""
    demo_dir = os.path.join(config.DATA_DIR, "demo")
    
    for i, class_name in enumerate(["texture_a", "texture_b", "texture_c"]):
        class_dir = os.path.join(demo_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for j in range(5):
            img = Image.new("RGB", (224, 224),
                          color=(i * 80 + j * 10, (i + 1) * 60, j * 50))
            img.save(os.path.join(class_dir, f"sample_{j}.jpg"))


if __name__ == "__main__":
    run_demo()
