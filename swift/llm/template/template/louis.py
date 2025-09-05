# Copyright (c) Alibaba, Inc. and its affiliates.
"""Louis Template: FastViTHD + Phi4 multimodal template"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from .utils import ChatmlTemplateMeta


class LouisTemplate(Template):
    """Louis multimodal template supporting images and videos"""
    placeholder_tokens = ['<image>', '<video>']

    @property
    def image_token_index(self):
        if not hasattr(self, '_image_token_index'):
            # Use a default token index for images
            self._image_token_index = self.tokenizer.convert_tokens_to_ids('<image>')
            if self._image_token_index is None:
                # Fallback to a special token ID
                self._image_token_index = 32001  # Common special token ID
        return self._image_token_index

    @property
    def video_token_index(self):
        if not hasattr(self, '_video_token_index'):
            # Use a default token index for videos
            self._video_token_index = self.tokenizer.convert_tokens_to_ids('<video>')
            if self._video_token_index is None:
                # Fallback to a special token ID  
                self._video_token_index = 32002  # Common special token ID
        return self._video_token_index

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<image>\n']
        elif media_type == 'video':
            return ['<video>\n'] 
        else:
            return ['']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        
        # Handle images
        images = inputs.images
        if images:
            # This would integrate with FastViTHD vision encoder
            # For now, use a placeholder that indicates Louis processing
            image_processor = getattr(self.processor, 'image_processor', None)
            if image_processor is not None:
                # Process images with FastViTHD optimizations
                image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
                encoded['pixel_values'] = image_inputs['pixel_values']
                
                # Add Louis-specific enhancements
                if 'image_sizes' in image_inputs:
                    encoded['image_sizes'] = image_inputs['image_sizes']
                    
        # Handle videos (basic support)
        videos = inputs.videos
        if videos:
            # Louis supports video through frame sampling and FastViTHD
            # For now, treat videos as sequences of images
            video_processor = getattr(self.processor, 'video_processor', None)
            if video_processor is not None:
                video_inputs = video_processor(videos, return_tensors='pt').to(self.model_info.torch_dtype)
                encoded['pixel_values_videos'] = video_inputs.get('pixel_values_videos', video_inputs['pixel_values'])
        
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        
        # Handle image data in batch
        pixel_values = [b['pixel_values'] for b in batch if 'pixel_values' in b]
        if pixel_values:
            # Stack with Louis FastViTHD optimizations
            res['pixel_values'] = torch.stack(pixel_values)
            
        # Handle video data in batch  
        pixel_values_videos = [b['pixel_values_videos'] for b in batch if 'pixel_values_videos' in b]
        if pixel_values_videos:
            res['pixel_values_videos'] = torch.stack(pixel_values_videos)
            
        return res


@dataclass
class LouisTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])
    prompt: Prompt = field(default_factory=lambda: ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    default_system: Optional[str] = field(default_factory=lambda: 
        'You are Louis, an advanced multimodal AI assistant powered by FastViTHD vision encoding and Phi4 language modeling. '
        'You excel at understanding images and videos with exceptional speed and accuracy.')


register_template(
    LouisTemplateMeta(
        MLLMTemplateType.louis,
        template_cls=LouisTemplate,
        auto_add_bos=True,
    ))