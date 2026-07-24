#!/usr/bin/env python3
"""
STAC: Spiking Transformer for Conversational AI
Copyright (C) 2024 STAC Authors

Licensed under the MIT License. See LICENSE file for details.

SpikingJelly Compatibility Layer
Provides cross-version compatibility for SpikingJelly components.
"""
import importlib.metadata
import logging
from packaging.version import parse
import torch

_logger = logging.getLogger(__name__)

try:
    SJ_VERSION = importlib.metadata.version("spikingjelly")
except Exception:
    SJ_VERSION = "0.0.0.0.14"

def get_neuron():
    from spikingjelly.activation_based.neuron import LIFNode
    return LIFNode

def get_converter():
    # Use a proper version comparison; string comparison is lexicographic and
    # would order e.g. "0.0.0.0.9" after "0.0.0.0.14".
    if parse(SJ_VERSION) >= parse("0.0.0.0.14"):
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
    """
    Per-tensor symmetric k-bit weight quantization.

    NOTE: activation quantization is NOT implemented; `n_bits_a` is accepted for API
    compatibility and ignored (a warning is logged when it is set).
    """

    def __init__(self, n_bits_w=8, n_bits_a=8):
        self.n_bits_w = n_bits_w
        self.n_bits_a = n_bits_a

    def __call__(self, model):
        """Apply quantization to model weights"""
        # Use k-bit quantization functions from spikingjelly
        return self._quantize_model(model)

    def _quantize_model(self, model):
        # Import quantize module inside the method to avoid circular imports
        from spikingjelly.activation_based import quantize

        if not hasattr(quantize, "k_bit_quantize"):
            raise RuntimeError(
                "spikingjelly.activation_based.quantize.k_bit_quantize is not available in the "
                f"installed SpikingJelly version ({SJ_VERSION}). Cannot apply k-bit quantization. "
                "Upgrade SpikingJelly (pip install spikingjelly -U --pre) or run without quantization."
            )

        if self.n_bits_a is not None:
            _logger.warning(
                "Quantizer(n_bits_a=%s): activation quantization is not implemented; "
                "only weights are quantized.", self.n_bits_a
            )

        # Apply quantization to model parameters.
        #
        # k_bit_quantize is DoReFa-style: it rounds onto a FIXED grid of 1/(2^k - 1),
        # which assumes inputs in [0, 1]. Feeding raw weights meant the step size was
        # 1/255 no matter how small the tensor was, so a "8-bit" pass over typical
        # transformer weights (std ~0.02, range ~±0.08) produced only ~40 distinct
        # levels — about 5 effective bits — and a 2.5% relative error. Scaling each
        # tensor to [-1, 1] first gives the requested resolution across its own range.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                scale = param.data.abs().max()
                if not torch.isfinite(scale) or scale == 0:
                    continue
                normalized = param.data / scale
                param.data = quantize.k_bit_quantize(normalized, k=self.n_bits_w) * scale
        return model

def get_quantizer():
    """Get Quantizer class for SpikingJelly 0.0.0.0.14"""
    # Use our custom implementation since the Quantizer class is not available 
    # in the specified SpikingJelly version 0.0.0.0.14
    return Quantizer

def get_surrogate():
    from spikingjelly.activation_based import surrogate
    return surrogate 