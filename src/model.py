"""
AuthNet Model Architecture
EmbeddingNet: ResNet-18 backbone → 128-dim L2-normalized embeddings.
Designed for deep metric learning (Triplet/Siamese training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class EmbeddingNet(nn.Module):
    """
    Deep metric learning model that maps images to a compact embedding space.
    
    Architecture:
        Input (3 × 224 × 224)
            → ResNet-18 backbone (pre-trained, early layers frozen)
            → 512-dim features
            → Embedding head: Linear(512→256) → BN → ReLU → Linear(256→128)
            → L2 normalization
            → 128-dim unit embedding ("fingerprint")
    """
    
    def __init__(
        self,
        embedding_dim: int = config.EMBEDDING_DIM,
        backbone_name: str = config.BACKBONE,
        pretrained: bool = config.PRETRAINED,
        freeze_layers: list = None,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        freeze_layers = freeze_layers or config.FREEZE_LAYERS
        
        # ── Backbone ──
        if backbone_name == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            backbone_dim = 512
        elif backbone_name == "resnet34":
            backbone = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            backbone_dim = 512
        elif backbone_name == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Remove the final FC layer — we only need feature extraction
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Store layer references for freezing and Grad-CAM
        self.layer4 = backbone.layer4
        
        # ── Embedding Head ──
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, embedding_dim),
        )
        
        # ── Freeze early layers ──
        self._freeze_layers(freeze_layers)
        
        # Store backbone feature dim for external access
        self.backbone_dim = backbone_dim
    
    def _freeze_layers(self, layer_names: list):
        """Freeze specified backbone layers to prevent overfitting."""
        for name, param in self.backbone.named_parameters():
            for freeze_name in layer_names:
                # Map layer names: layer1 -> index 4, layer2 -> index 5
                layer_map = {
                    "conv1": "0.", "bn1": "1.", 
                    "layer1": "4.", "layer2": "5.",
                }
                if freeze_name in layer_map and name.startswith(layer_map[freeze_name]):
                    param.requires_grad = False
                    break
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone (before embedding head)."""
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → L2-normalized embedding.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            L2-normalized embedding of shape (batch_size, embedding_dim)
        """
        features = self.get_backbone_features(x)
        embeddings = self.embedding_head(features)
        
        # L2 normalize — critical for cosine similarity to equal dot product
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() — more descriptive name for inference."""
        return self.forward(x)
    
    def get_parameter_groups(self) -> list:
        """
        Return parameter groups with differential learning rates.
        Backbone gets lower LR (fine-tuning), head gets higher LR (training from scratch).
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.embedding_head.parameters())
        
        return [
            {"params": backbone_params, "lr": config.LR_BACKBONE},
            {"params": head_params, "lr": config.LR_HEAD},
        ]
    
    def count_parameters(self) -> dict:
        """Count total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "total_mb": total * 4 / (1024 ** 2),  # FP32
        }


def build_model(device: torch.device = None) -> EmbeddingNet:
    """Build and return the EmbeddingNet model on the specified device."""
    device = device or config.DEVICE
    model = EmbeddingNet(
        embedding_dim=config.EMBEDDING_DIM,
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
    )
    model = model.to(device)
    
    param_info = model.count_parameters()
    print(f"\nModel: EmbeddingNet ({config.BACKBONE} -> {config.EMBEDDING_DIM}-dim)")
    print(f"  Total params:     {param_info['total']:,}")
    print(f"  Trainable params: {param_info['trainable']:,}")
    print(f"  Frozen params:    {param_info['frozen']:,}")
    print(f"  Model size:       {param_info['total_mb']:.1f} MB (FP32)")
    print(f"  Device:           {device}\n")
    
    return model


def load_model(checkpoint_path: str, device: torch.device = None) -> EmbeddingNet:
    """Load a trained model from checkpoint."""
    device = device or config.DEVICE
    model = EmbeddingNet(
        embedding_dim=config.EMBEDDING_DIM,
        backbone_name=config.BACKBONE,
        pretrained=False,  # We'll load our own weights
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Handle both full checkpoint and state_dict-only saves
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    return model
