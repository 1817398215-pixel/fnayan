# models.py
# 模型统一导入模块

from base_components import Residual, PreNorm, FeedForward, Attention, Transformer
from spectral_sequence_models import SpectralSeqFusion

__all__ = [
    # Base components
    'Residual',
    'PreNorm', 
    'FeedForward',
    'Attention',
    'Transformer',
    
    # Fusion models
    'SpectralSeqFusion'
]