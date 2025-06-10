#!/usr/bin/env python3
"""
STAC: Spiking Transformer for Conversational AI
Copyright (C) 2024 STAC Authors

Licensed under the MIT License. See LICENSE file for details.

SpikingJelly Compatibility Layer
Provides cross-version compatibility for SpikingJelly components.
"""
import importlib.metadata
from packaging.version import parse
import torch

try:
    SJ_VERSION = importlib.metadata.version("spikingjelly")
except:
    SJ_VERSION = "0.0.0.0.14"

def get_neuron():
    from spikingjelly.activation_based.neuron import LIFNode
    return LIFNode

def get_converter():
    if SJ_VERSION >= "0.0.0.0.14":
        try:
            from spikingjelly.activation_based.conversion import Converter
            return Converter
        except ImportError:
            from spikingjelly.activation_based.ann2snn import Converter
            return Converter
    else:
        from spikingjelly.activation_based.ann2snn import Converter
        return Converter

# Custom Quantizer class implementation since it's not available in the installed version
class Quantizer:
    def __init__(self, n_bits_w=8, n_bits_a=8):
        self.n_bits_w = n_bits_w
        self.n_bits_a = n_bits_a
        
    def __call__(self, model):
        """Apply quantization to model weights and activations"""
        # Use k-bit quantization functions from spikingjelly
        return self._quantize_model(model)
    
    def _quantize_model(self, model):
        # Import quantize module inside the method to avoid circular imports
        from spikingjelly.activation_based import quantize
        
        # Apply quantization to model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Apply k-bit quantization to weights
                param.data = quantize.k_bit_quantize(param.data, k=self.n_bits_w)
        return model

def get_quantizer():
    """Get Quantizer class for SpikingJelly 0.0.0.0.14"""
    # Use our custom implementation since the Quantizer class is not available 
    # in the specified SpikingJelly version 0.0.0.0.14
    return Quantizer

def get_surrogate():
    from spikingjelly.activation_based import surrogate
    return surrogate 