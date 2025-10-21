"""
RVSP Frequency Optimizer Project
================================

This package brings together:
- The model architecture (ComplexFFNN)
- The custom optimizer (custom_adam_adaptive_momentum)
- The MNIST training and evaluation pipeline

Usage examples:
---------------
from src.models import ComplexFFNN
from src.optimizers import custom_adam_adaptive_momentum
from src.training.train_mnist import train_with_accuracy
"""

from src.models.complex_ffnn import ComplexFFNN
from src.optimizers.custom_adam_adaptive_momentum import custom_adam_adaptive_momentum
