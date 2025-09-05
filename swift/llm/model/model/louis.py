# Copyright (c) Alibaba, Inc. and its affiliates.
"""Louis Model: FastViTHD Vision Encoder + Phi4 LLM"""

import os
import sys
from functools import partial
from typing import Any, Dict

from swift.llm import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal, register_model)
from ..utils import ModelInfo


def get_model_tokenizer_louis(model_dir: str,
                             model_info: ModelInfo,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             **kwargs):
    """Get Louis model and tokenizer"""
    from transformers import AutoConfig, AutoTokenizer
    from ..multimodal_encoder.fastvit_hd_encoder import FastViTHDVisionTower
    
    # Load Phi4 as the base language model
    kwargs['automodel_class'] = None  # Let transformers auto-detect
    
    # Set up vision tower configuration
    vision_tower_path = kwargs.get('vision_tower_path', 'openai/clip-vit-large-patch14-336')
    
    # Mock a simple Louis configuration for now
    class LouisConfig:
        def __init__(self):
            self.mm_vision_tower = vision_tower_path
            self.mm_vision_select_layer = -1
            self.mm_vision_select_feature = 'patch'
            self.mm_projector_type = 'linear'
            self.mm_hidden_size = 1024
            
    # For now, load as a regular multimodal model and customize later
    model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)
    
    # TODO: Replace the vision tower with FastViTHD
    if model is not None and load_model:
        # This would be where we integrate FastViTHD
        # For now, we'll use the existing vision tower but mark it for future enhancement
        pass
        
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.louis,
        [
            ModelGroup([
                # For now, we'll use existing models as placeholders
                # In a real implementation, these would be trained Louis models
                Model('microsoft/phi-4', 'microsoft/phi-4'),  # Base Phi4 model
            ]),
        ],
        TemplateType.louis,
        get_model_tokenizer_louis,
        architectures=['Phi3ForCausalLM'],  # Will be enhanced to LouisForCausalLM
        model_arch=ModelArch.louis,
        requires=['transformers>=4.36'],
        tags=['vision', 'video'],
    ))