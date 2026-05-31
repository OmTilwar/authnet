"""
Tests for EmbeddingNet model architecture.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np


class TestEmbeddingNet:
    """Test suite for the EmbeddingNet model."""
    
    def test_model_instantiation(self):
        """Test model can be created with default config."""
        from src.model import EmbeddingNet
        model = EmbeddingNet(embedding_dim=128, backbone_name="resnet18", pretrained=False)
        assert model is not None
    
    def test_output_shape(self):
        """Test output embedding has correct shape (batch_size, 128)."""
        from src.model import EmbeddingNet
        model = EmbeddingNet(embedding_dim=128, backbone_name="resnet18", pretrained=False)
        model.eval()
        
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 128), f"Expected (4, 128), got {output.shape}"
    
    def test_l2_normalization(self):
        """Test all output embeddings have unit L2 norm."""
        from src.model import EmbeddingNet
        model = EmbeddingNet(embedding_dim=128, backbone_name="resnet18", pretrained=False)
        model.eval()
        
        x = torch.randn(8, 3, 224, 224)
        
        with torch.no_grad():
            embeddings = model(x)
        
        norms = torch.norm(embeddings, p=2, dim=1)
        
        # All norms should be approximately 1.0
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Embeddings not normalized. Norms: {norms}"
    
    def test_different_embedding_dims(self):
        """Test model works with different embedding dimensions."""
        from src.model import EmbeddingNet
        
        for dim in [64, 128, 256, 512]:
            model = EmbeddingNet(embedding_dim=dim, backbone_name="resnet18", pretrained=False)
            model.eval()
            
            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, dim), f"Expected (2, {dim}), got {output.shape}"
    
    def test_frozen_layers(self):
        """Test that specified layers are actually frozen."""
        from src.model import EmbeddingNet
        model = EmbeddingNet(
            embedding_dim=128, backbone_name="resnet18", 
            pretrained=False, freeze_layers=["layer1", "layer2"]
        )
        
        # Check that some parameters are frozen
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen_count > 0, "No parameters are frozen"
    
    def test_parameter_groups(self):
        """Test differential learning rate parameter groups."""
        from src.model import EmbeddingNet
        model = EmbeddingNet(embedding_dim=128, backbone_name="resnet18", pretrained=False)
        
        param_groups = model.get_parameter_groups()
        
        assert len(param_groups) == 2, "Expected 2 parameter groups (backbone + head)"
        assert 'lr' in param_groups[0], "Backbone group missing lr"
        assert 'lr' in param_groups[1], "Head group missing lr"
        assert param_groups[0]['lr'] < param_groups[1]['lr'], \
            "Backbone lr should be lower than head lr"
    
    def test_count_parameters(self):
        """Test parameter counting utility."""
        from src.model import EmbeddingNet
        model = EmbeddingNet(embedding_dim=128, backbone_name="resnet18", pretrained=False)
        
        counts = model.count_parameters()
        
        assert counts['total'] > 0
        assert counts['trainable'] > 0
        assert counts['trainable'] <= counts['total']
        assert counts['total_mb'] > 0
    
    def test_deterministic_output(self):
        """Test model gives same output for same input (eval mode)."""
        from src.model import EmbeddingNet
        model = EmbeddingNet(embedding_dim=128, backbone_name="resnet18", pretrained=False)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        assert torch.allclose(out1, out2, atol=1e-6), "Outputs should be deterministic in eval mode"
    
    def test_get_backbone_features(self):
        """Test backbone feature extraction before embedding head."""
        from src.model import EmbeddingNet
        model = EmbeddingNet(embedding_dim=128, backbone_name="resnet18", pretrained=False)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            features = model.get_backbone_features(x)
        
        assert features.shape == (2, 512), f"Expected (2, 512), got {features.shape}"
