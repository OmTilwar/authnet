"""
AuthNet Fingerprinting Module
Extracts visual fingerprints (embeddings) and performs authentication via similarity matching.
Maps directly to Entrupy's core products:
  - create_fingerprint() → Entrupy Fingerprinting
  - authenticate() → Entrupy Authentication
"""

import os
import sys
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model import EmbeddingNet, load_model
from src.dataset import get_test_transforms


class FingerprintEngine:
    """
    Product authentication and fingerprinting engine.
    
    Uses the trained EmbeddingNet to:
    1. Generate unique visual fingerprints (128-dim embeddings)
    2. Authenticate items by comparing against a gallery of known items
    3. Verify if two items are the same
    """
    
    def __init__(
        self,
        model: Optional[EmbeddingNet] = None,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Pre-loaded EmbeddingNet model
            model_path: Path to model checkpoint (used if model is None)
            device: Torch device
        """
        self.device = device or config.DEVICE
        self.transform = get_test_transforms()
        
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        elif model_path and os.path.exists(model_path):
            self.model = load_model(model_path, self.device)
        else:
            raise ValueError("Either 'model' or a valid 'model_path' must be provided.")
        
        # Gallery storage
        self.gallery_embeddings: Optional[np.ndarray] = None
        self.gallery_labels: Optional[List[str]] = None
        self.gallery_paths: Optional[List[str]] = None
    
    def _preprocess_image(self, image_input) -> torch.Tensor:
        """
        Preprocess an image for the model.
        
        Args:
            image_input: PIL Image, numpy array (HWC, BGR or RGB), or file path string
        
        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224)
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Assume BGR (OpenCV) → convert to RGB PIL
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image = Image.fromarray(image_input[:, :, ::-1])
            else:
                image = Image.fromarray(image_input)
            image = image.convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
        
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    @torch.no_grad()
    def create_fingerprint(self, image_input) -> np.ndarray:
        """
        Generate a unique visual fingerprint for an image.
        
        Args:
            image_input: Image path, PIL Image, or numpy array
            
        Returns:
            128-dim numpy array (the fingerprint)
        """
        tensor = self._preprocess_image(image_input)
        embedding = self.model(tensor)
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def create_fingerprints_batch(self, image_inputs: list) -> np.ndarray:
        """
        Generate fingerprints for a batch of images.
        
        Args:
            image_inputs: List of image paths, PIL Images, or numpy arrays
            
        Returns:
            (N, 128) numpy array of fingerprints
        """
        tensors = []
        for img in image_inputs:
            tensors.append(self._preprocess_image(img))
        
        batch = torch.cat(tensors, dim=0)
        embeddings = self.model(batch)
        return embeddings.cpu().numpy()
    
    def build_gallery(
        self,
        image_dir: str,
        save_path: Optional[str] = None,
    ) -> int:
        """
        Build a fingerprint gallery from a directory of images.
        
        Args:
            image_dir: Directory with images (can have class subdirectories)
            save_path: Where to save the gallery (.npz file)
        
        Returns:
            Number of images in the gallery
        """
        save_path = save_path or os.path.join(config.EMBED_DIR, "gallery.npz")
        
        embeddings = []
        labels = []
        paths = []
        
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Check if class-based subdirectory structure
        subdirs = [d for d in os.listdir(image_dir)
                    if os.path.isdir(os.path.join(image_dir, d))]
        
        if subdirs:
            # Class-based structure
            for class_name in tqdm(sorted(subdirs), desc="Building gallery"):
                class_path = os.path.join(image_dir, class_name)
                for img_name in os.listdir(class_path):
                    if os.path.splitext(img_name)[1].lower() not in valid_ext:
                        continue
                    
                    img_path = os.path.join(class_path, img_name)
                    fp = self.create_fingerprint(img_path)
                    
                    embeddings.append(fp)
                    labels.append(class_name)
                    paths.append(img_path)
        else:
            # Flat structure
            for img_name in tqdm(sorted(os.listdir(image_dir)), desc="Building gallery"):
                if os.path.splitext(img_name)[1].lower() not in valid_ext:
                    continue
                
                img_path = os.path.join(image_dir, img_name)
                fp = self.create_fingerprint(img_path)
                
                embeddings.append(fp)
                labels.append("unknown")
                paths.append(img_path)
        
        self.gallery_embeddings = np.array(embeddings)
        self.gallery_labels = labels
        self.gallery_paths = paths
        
        # Save gallery
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            embeddings=self.gallery_embeddings,
            labels=np.array(labels),
            paths=np.array(paths),
        )
        
        print(f"Gallery built: {len(embeddings)} items, saved to {save_path}")
        return len(embeddings)
    
    def load_gallery(self, gallery_path: str):
        """Load a pre-built gallery from disk."""
        data = np.load(gallery_path, allow_pickle=True)
        self.gallery_embeddings = data['embeddings']
        self.gallery_labels = data['labels'].tolist()
        self.gallery_paths = data['paths'].tolist()
        
        print(f"Gallery loaded: {len(self.gallery_embeddings)} items from {gallery_path}")
    
    def authenticate(
        self,
        query_image,
        threshold: float = None,
        top_k: int = 5,
    ) -> Dict:
        """
        Authenticate an item by comparing against the gallery.
        
        Args:
            query_image: Image to authenticate
            threshold: Cosine similarity threshold (default: config.AUTH_THRESHOLD)
            top_k: Number of top matches to return
        
        Returns:
            dict with:
                - is_authentic: bool
                - confidence: float (max similarity score)
                - best_match: str (label of closest match)
                - best_match_path: str (path of closest match)
                - top_k_matches: list of (label, score, path)
                - verdict: str ("AUTHENTIC" / "SUSPICIOUS" / "UNKNOWN")
        """
        if self.gallery_embeddings is None:
            raise RuntimeError("No gallery loaded. Call build_gallery() or load_gallery() first.")
        
        threshold = threshold or config.AUTH_THRESHOLD
        
        # Get query fingerprint
        query_fp = self.create_fingerprint(query_image)
        
        # Compute cosine similarities
        similarities = np.dot(self.gallery_embeddings, query_fp)
        
        # Get top-K matches
        top_k_idx = np.argsort(-similarities)[:top_k]
        
        top_k_matches = [
            {
                'label': self.gallery_labels[i],
                'similarity': float(similarities[i]),
                'path': self.gallery_paths[i],
            }
            for i in top_k_idx
        ]
        
        best_idx = top_k_idx[0]
        best_similarity = float(similarities[best_idx])
        best_label = self.gallery_labels[best_idx]
        
        # Determine verdict
        if best_similarity >= threshold:
            verdict = "AUTHENTIC"
            is_authentic = True
        elif best_similarity >= threshold * 0.8:
            verdict = "SUSPICIOUS"
            is_authentic = False
        else:
            verdict = "NO_MATCH"
            is_authentic = False
        
        return {
            'is_authentic': is_authentic,
            'confidence': best_similarity,
            'best_match': best_label,
            'best_match_path': self.gallery_paths[best_idx],
            'top_k_matches': top_k_matches,
            'verdict': verdict,
            'threshold': threshold,
        }
    
    def verify_pair(
        self,
        image_a,
        image_b,
        threshold: float = None,
    ) -> Dict:
        """
        Verify if two images depict the same item.
        
        Args:
            image_a: First image
            image_b: Second image
            threshold: Similarity threshold
        
        Returns:
            dict with similarity score and match verdict
        """
        threshold = threshold or config.AUTH_THRESHOLD
        
        fp_a = self.create_fingerprint(image_a)
        fp_b = self.create_fingerprint(image_b)
        
        similarity = float(np.dot(fp_a, fp_b))
        
        return {
            'is_match': similarity >= threshold,
            'similarity': similarity,
            'threshold': threshold,
            'verdict': "MATCH" if similarity >= threshold else "MISMATCH",
        }
