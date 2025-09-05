# Copyright (c) Alibaba, Inc. and its affiliates.
"""FastViTHD Vision Encoder for Louis model"""

import os
from typing import Optional, Union, List
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, PreTrainedModel


class FastViTHDVisionTower(nn.Module):
    """
    FastViTHD Vision Tower - A hybrid vision encoder designed for efficient encoding of high-resolution images.
    This is a simplified implementation based on the Apple FastViT architecture.
    """

    def __init__(self, vision_tower_path: str, args, **kwargs):
        super().__init__()
        
        self.is_loaded = False
        self.vision_tower_path = vision_tower_path
        self.args = args
        
        # For now, we'll use CLIP as the base and add FastViT optimizations later
        # This allows us to get the basic structure working first
        self.vision_tower_name = vision_tower_path
        self.select_layer = getattr(args, 'mm_vision_select_layer', -1)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
    def load_model(self, device_map=None):
        if self.is_loaded:
            return
            
        # Load base CLIP model - we'll enhance this with FastViT features
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_path, 
            torch_dtype=torch.float16
        )
        
        # Create image processor
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_path
        )
        
        # Add FastViT-like optimizations
        self._add_fastvit_optimizations()
        
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def _add_fastvit_optimizations(self):
        """Add FastViT-like optimizations for efficiency"""
        # FastViT uses hybrid tokens - we'll simulate this with adaptive pooling
        # and token reduction strategies
        
        original_config = self.vision_tower.config
        self.hidden_size = original_config.hidden_size
        
        # Add adaptive pooling for token reduction (FastViT key feature)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))  # Reduce to 256 tokens max
        
        # Add lightweight MLP for feature enhancement
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
            nn.Dropout(0.1)
        )
        
    def feature_select(self, image_forward_outs):
        """Select features from vision encoder outputs"""
        image_features = image_forward_outs.hidden_states[self.select_layer]
        
        # Apply FastViT-like optimizations
        if self.select_feature == 'patch':
            # Remove CLS token if present
            if image_features.shape[1] > 1:
                image_features = image_features[:, 1:]  # Remove CLS token
                
        # Apply adaptive pooling for token reduction (FastViT feature)
        B, N, C = image_features.shape
        H = W = int(N ** 0.5)  # Assume square image patches
        
        if H * W == N:
            image_features = image_features.reshape(B, H, W, C).permute(0, 3, 1, 2)
            # Apply adaptive pooling to reduce tokens
            image_features = self.adaptive_pool(image_features)
            image_features = image_features.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        # Apply feature enhancement
        image_features = self.feature_enhancer(image_features)
        
        return image_features

    def forward(self, images):
        if not self.is_loaded:
            self.load_model()
            
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.unsqueeze(0).to(self.vision_tower.device), 
                                                    output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(self.vision_tower.device), 
                                                 output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, dtype=self.vision_tower.dtype, 
                          device=self.vision_tower.device)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self._hidden_size if hasattr(self, '_hidden_size') else 1024

    @hidden_size.setter
    def hidden_size(self, value):
        self._hidden_size = value

    @property
    def num_patches_per_side(self):
        return self.image_processor.crop_size["height"] // self.image_processor.crop_size["width"]

    @property
    def num_patches(self):
        # After adaptive pooling, we have 16x16 = 256 patches max
        return 256