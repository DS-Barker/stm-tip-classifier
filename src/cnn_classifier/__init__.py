# src/cnn_classifier/__init__.py
"""
CNN Classifier Module

Provides model architectures, training, and evaluation for STM tip classification.
"""

from .model import build_model, get_thesis_architecture
from .train import train_single_model
from .evaluate import evaluate_model

__all__ = [
    'build_model',
    'get_thesis_architecture',
    'train_single_model',
    'evaluate_model'
]