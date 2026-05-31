"""
Tests for dataset pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np
import tempfile
from PIL import Image


def create_temp_dataset(num_classes=5, images_per_class=10, image_size=64):
    """Create a temporary dataset directory with synthetic images."""
    temp_dir = tempfile.mkdtemp()
    
    for i in range(num_classes):
        class_dir = os.path.join(temp_dir, f"class_{i:03d}")
        os.makedirs(class_dir)
        
        for j in range(images_per_class):
            # Create a random colored image (different color per class for testing)
            img = Image.new("RGB", (image_size, image_size), 
                          color=(i * 50 % 256, j * 25 % 256, (i + j) * 30 % 256))
            img.save(os.path.join(class_dir, f"img_{j:04d}.jpg"))
    
    return temp_dir


class TestTextureDataset:
    """Test suite for the TextureDataset class."""
    
    def test_dataset_loading(self):
        """Test dataset loads images correctly."""
        from src.dataset import TextureDataset
        
        temp_dir = create_temp_dataset(num_classes=3, images_per_class=5)
        dataset = TextureDataset(temp_dir, min_samples_per_class=2)
        
        assert len(dataset) == 15, f"Expected 15 images, got {len(dataset)}"
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_image_tensor_shape(self):
        """Test returned images have correct tensor shape."""
        from src.dataset import TextureDataset, get_test_transforms
        
        temp_dir = create_temp_dataset(num_classes=2, images_per_class=3)
        dataset = TextureDataset(temp_dir, transform=get_test_transforms())
        
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224), f"Expected (3, 224, 224), got {image.shape}"
        assert isinstance(label, int)
        
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_value_range(self):
        """Test image tensors are properly normalized."""
        from src.dataset import TextureDataset, get_test_transforms
        
        temp_dir = create_temp_dataset(num_classes=2, images_per_class=3)
        dataset = TextureDataset(temp_dir, transform=get_test_transforms())
        
        image, _ = dataset[0]
        
        # After ImageNet normalization, values can be outside [0, 1]
        # but should be within a reasonable range
        assert image.min() > -5.0, f"Suspiciously low value: {image.min()}"
        assert image.max() < 5.0, f"Suspiciously high value: {image.max()}"
        
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_min_samples_filter(self):
        """Test classes with too few samples are filtered out."""
        from src.dataset import TextureDataset
        
        temp_dir = create_temp_dataset(num_classes=3, images_per_class=5)
        
        # With min_samples=10, only classes with 10+ samples should be included
        # Our test classes have 5 each, so none should pass
        dataset = TextureDataset(temp_dir, min_samples_per_class=10)
        assert len(dataset) == 0
        
        # With min_samples=2, all should pass
        dataset = TextureDataset(temp_dir, min_samples_per_class=2)
        assert len(dataset) == 15
        
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_get_labels(self):
        """Test get_labels returns all labels."""
        from src.dataset import TextureDataset
        
        temp_dir = create_temp_dataset(num_classes=4, images_per_class=5)
        dataset = TextureDataset(temp_dir, min_samples_per_class=2)
        
        labels = dataset.get_labels()
        
        assert len(labels) == 20
        assert len(set(labels)) == 4  # 4 unique classes
        
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_augmentation_produces_different_outputs(self):
        """Test that training augmentations produce different outputs for same image."""
        from src.dataset import TextureDataset, get_train_transforms
        
        # Create a textured image (gradient) instead of solid color
        # Solid colors are invariant to geometric transforms after normalization
        temp_dir = tempfile.mkdtemp()
        class_dir = os.path.join(temp_dir, "textured_class")
        os.makedirs(class_dir)
        
        img_array = np.zeros((128, 128, 3), dtype=np.uint8)
        for i in range(128):
            for j in range(128):
                img_array[i, j] = [(i * 2) % 256, (j * 2) % 256, ((i + j) * 3) % 256]
        Image.fromarray(img_array).save(os.path.join(class_dir, "textured.jpg"))
        
        dataset = TextureDataset(temp_dir, transform=get_train_transforms(), min_samples_per_class=1)
        
        # Get same image twice — augmentation should make them different
        img1, _ = dataset[0]
        img2, _ = dataset[0]
        
        # They should NOT be identical (randomized augmentation on a textured image)
        assert not torch.allclose(img1, img2, atol=1e-3), \
            "Augmented images should differ"
        
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_inverse_normalize(self):
        """Test inverse normalization produces valid image."""
        from src.dataset import TextureDataset, get_test_transforms, inverse_normalize
        
        temp_dir = create_temp_dataset(num_classes=1, images_per_class=1)
        dataset = TextureDataset(temp_dir, transform=get_test_transforms(), min_samples_per_class=1)
        
        tensor, _ = dataset[0]
        
        # Inverse normalize
        image = inverse_normalize(tensor)
        
        assert image.shape == (224, 224, 3)
        assert image.dtype == np.uint8
        assert image.min() >= 0
        assert image.max() <= 255
        
        import shutil
        shutil.rmtree(temp_dir)
