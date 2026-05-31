"""
AuthNet Dataset Pipeline
Loads DTD + MVTec AD texture datasets for metric learning training.
Uses pytorch-metric-learning compatible format: returns (image, label) pairs.
"""

import os
import random
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def get_train_transforms() -> transforms.Compose:
    """Training augmentation pipeline — texture-aware augmentations."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_test_transforms() -> transforms.Compose:
    """Test/inference transforms — no augmentation."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class TextureDataset(Dataset):
    """
    Generic texture dataset that loads images from class-based subdirectories.
    
    Expected structure:
        root_dir/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg
                ...
    
    Returns (image_tensor, class_label) pairs compatible with 
    pytorch-metric-learning miners and losses.
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        min_samples_per_class: int = 2,
    ):
        """
        Args:
            root_dir: Path to dataset root with class subdirectories.
            transform: Torchvision transforms to apply.
            min_samples_per_class: Minimum images per class to include 
                                   (classes with fewer are skipped).
        """
        self.root_dir = root_dir
        self.transform = transform or get_test_transforms()
        
        # Build image list and class mapping
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        if not os.path.exists(root_dir):
            print(f"[WARN] Dataset directory not found: {root_dir}")
            return
        
        # Scan classes
        class_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        class_idx = 0
        
        for class_name in class_dirs:
            class_path = os.path.join(root_dir, class_name)
            images = [
                f for f in os.listdir(class_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            
            # Skip classes with too few samples
            if len(images) < min_samples_per_class:
                continue
            
            self.class_names.append(class_name)
            self.class_to_idx[class_name] = class_idx
            
            for img_name in sorted(images):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(class_idx)
            
            class_idx += 1
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load with PIL (more robust than OpenCV for various formats)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback: return a random valid image from same class
            print(f"[WARN] Failed to load {img_path}: {e}")
            same_class = [i for i, l in enumerate(self.labels) if l == label and i != idx]
            if same_class:
                return self.__getitem__(random.choice(same_class))
            # Last resort: return blank image
            image = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_labels(self) -> List[int]:
        """Return all labels — needed by MPerClassSampler."""
        return self.labels
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from label index."""
        if idx < len(self.class_names):
            return self.class_names[idx]
        return f"class_{idx}"
    
    def summary(self) -> str:
        """Print dataset summary."""
        n_classes = len(self.class_names)
        n_images = len(self.image_paths)
        if n_images == 0:
            return f"Empty dataset at {self.root_dir}"
        
        samples_per_class = {}
        for label in self.labels:
            samples_per_class[label] = samples_per_class.get(label, 0) + 1
        
        counts = list(samples_per_class.values())
        return (
            f"Dataset: {self.root_dir}\n"
            f"  Classes: {n_classes}\n"
            f"  Images:  {n_images}\n"
            f"  Samples/class: min={min(counts)}, max={max(counts)}, "
            f"mean={sum(counts)/len(counts):.1f}"
        )


def create_dataloaders(
    train_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, TextureDataset, TextureDataset]:
    """
    Create train and test DataLoaders with MPerClassSampler.
    
    Returns:
        (train_loader, test_loader, train_dataset, test_dataset)
    """
    from pytorch_metric_learning.samplers import MPerClassSampler
    
    train_dir = train_dir or os.path.join(config.COMBINED_DIR, "train")
    test_dir = test_dir or os.path.join(config.COMBINED_DIR, "test")
    batch_size = batch_size or config.BATCH_SIZE
    
    # Create datasets
    train_dataset = TextureDataset(
        root_dir=train_dir,
        transform=get_train_transforms(),
        min_samples_per_class=config.SAMPLES_PER_CLASS,
    )
    
    test_dataset = TextureDataset(
        root_dir=test_dir,
        transform=get_test_transforms(),
        min_samples_per_class=2,
    )
    
    print(f"\n{'='*50}")
    print("Dataset Summary")
    print(f"{'='*50}")
    print(f"[Train] {train_dataset.summary()}")
    print(f"[Test]  {test_dataset.summary()}")
    print(f"{'='*50}\n")
    
    # MPerClassSampler ensures each batch has m samples per class
    # This is critical for triplet mining to find valid triplets within a batch
    train_labels = train_dataset.get_labels()
    
    # Adjust batch_size to be compatible with sampler
    m = config.SAMPLES_PER_CLASS
    n_classes_per_batch = batch_size // m
    effective_batch_size = n_classes_per_batch * m
    
    sampler = MPerClassSampler(
        labels=train_labels,
        m=m,
        batch_size=effective_batch_size,
        length_before_new_iter=len(train_dataset),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


def inverse_normalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor back to a displayable numpy image (H, W, C)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor.cpu().clone()
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy HWC format
    image = tensor.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return image
