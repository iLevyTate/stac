#!/usr/bin/env python3
"""
STAC: Spiking Transformer for Conversational AI
Copyright (C) 2024 STAC Authors

Licensed under the MIT License. See LICENSE file for details.

Test conversational capabilities of the SNN model.
Verifies that the model can maintain state between conversation turns.
"""
import os
import torch
import argparse
import logging
import sys
import tempfile
from pathlib import Path

# Allow running this file directly by putting the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer, AutoModelForCausalLM
from smollm2_converter import (
    replace_gelu_with_relu,
    simplified_conversion,
    apply_surrogate_gradients,
    calibrate_timesteps,
    save_snn_model,
    TemporalSpikeProcessor
)
import pytest
from _pytest.outcomes import Failed, Skipped
import torch.profiler

# Forwards per profiled measurement. Averaging over several calls keeps the wall-clock
# comparison from being dominated by one-off warmup cost.
PROFILE_REPEATS = 3

from loihi_constraints import validate_loihi_export_readiness, write_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('conversation_test.log')
    ],
    force=True
)
logger = logging.getLogger("conversation_test")
logger.info("Starting test_conversational_snn.py script...")


def _accelerator_time_total(evt) -> float:
    """
    Total accelerator (GPU) time for a profiler event, across PyTorch versions.

    `FunctionEventAvg.cuda_time_total` was renamed to `device_time_total` and later
    removed; reading it unconditionally raised AttributeError on modern PyTorch and
    made the whole energy test fail before it measured anything. Returns 0.0 when the
    profile contains no accelerator activity (e.g. CPU-only runs).
    """
    for attr in ("device_time_total", "cuda_time_total"):
        value = getattr(evt, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _safe_console_text(s: str) -> str:
    """
    Ensure text is safely printable on Windows consoles that may default to cp1252.
    This prevents logging from throwing UnicodeEncodeError on generated tokens.
    """
    try:
        return s.encode("cp1252", errors="backslashreplace").decode("cp1252")
    except Exception:
        # Last resort: replace anything non-ascii
        return "".join(ch if ord(ch) < 128 else "?" for ch in s)

def parse_args(argv=None):
    """Parse CLI arguments. `argv=None` reads sys.argv; pass a list to build defaults."""
    parser = argparse.ArgumentParser(description='Test SNN Conversational Pipeline')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Model name or path')
    parser.add_argument('--output_dir', type=str, default='./test_output',
                      help='Directory for test outputs')
    parser.add_argument('--timesteps', type=int, default=16,
                      help='Number of timesteps for SNN')
    parser.add_argument('--test_turns', type=int, default=3,
                      help='Number of conversation turns to test')
    parser.add_argument('--max_context_length', type=int, default=2048,
                      help='Maximum context length')
    # Several tests read args.device directly. It used to be attached only by main()
    # after parsing, so any other caller (pytest, a script importing these helpers) hit
    # AttributeError: 'Namespace' object has no attribute 'device'.
    parser.add_argument('--device', type=str, default=None, choices=[None, 'cpu', 'cuda'],
                      help='Device to run on (default: cuda when available, else cpu)')
    
    # Add test flags
    parser.add_argument('--test_all', action='store_true',
                      help='Run all tests')
    parser.add_argument('--test_position_boundaries', action='store_true',
                      help='Test position ID boundaries')
    parser.add_argument('--test_attention_mask', action='store_true',
                      help='Test attention mask continuity')
    parser.add_argument('--test_multi_turn', action='store_true',
                      help='Test multi-turn coherence')
    parser.add_argument('--test_energy', action='store_true',
                      help='Test energy consumption')
    parser.add_argument('--test_loihi_constraints', action='store_true',
                      help='Run Loihi export-readiness constraints validator (simulation-time)')
    parser.add_argument('--test_tsp_state', action='store_true',
                      help='Deterministic TemporalSpikeProcessor state retention test (incremental mode)')
    parser.add_argument('--test_fidelity', action='store_true',
                      help='ANN↔SNN fidelity parity test (logit diff + top-1 agreement) on fixed prompts')
    parser.add_argument('--test_parity_multi_turn', action='store_true',
                      help='Deterministic multi-turn parity test (greedy) comparing ANN vs SNN next-token trajectories')
    parser.add_argument('--parity_steps', type=int, default=16,
                      help='Number of next-token steps to compare in parity tests (default: 16)')
    parser.add_argument('--parity_tolerance', type=float, default=1e-2,
                      help='Max-abs logit tolerance for parity checks (default: 1e-2)')
    parser.add_argument('--fidelity_gate', type=str, default='either', choices=['either', 'logits', 'top1'],
                      help="Fidelity parity pass gate: 'logits' uses --parity_tolerance, 'top1' uses --fidelity_top1_min, 'either' accepts either.")
    parser.add_argument('--fidelity_top1_min', type=float, default=0.50,
                      help='Minimum top-1 match rate to pass fidelity parity when gate is top1/either (default: 0.50)')
    parser.add_argument('--adapter_dir', type=str, default=None,
                      help='Optional PEFT adapter directory to load into SNN inner model before running parity tests.')
    parser.add_argument('--loihi_mode', action='store_true',
                      help='Enable Loihi-oriented conversion mode (replaces attention with LoihiCausalContextMixer).')
    parser.add_argument('--loihi_quantize', action='store_true',
                      help='Enable fake int8 weight quantization pass (simulation-time, export-readiness evidence).')
    parser.add_argument('--loihi_prune', action='store_true',
                      help='Enable magnitude-based pruning pass (simulation-time sparsity evidence).')
    parser.add_argument('--loihi_prune_target', type=float, default=0.5,
                      help='Target sparsity for pruning (0-1). Default: 0.5')
    parser.add_argument('--loihi_embed_buckets', type=int, default=0,
                      help='If >0, replace token embedding table with hashed/bucketed embedding of this size (Loihi memory reduction).')
    parser.add_argument('--skip_gelu_replacement', action='store_true',
                      help='Skip GELU->ReLU replacement to preserve generation quality. Default: True for better coherence.')

    parsed = parser.parse_args(argv)
    if getattr(parsed, 'device', None) is None:
        parsed.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return parsed


# --- pytest fixtures ---------------------------------------------------------------
# The test functions in this file take (model, tokenizer, args)-style parameters because
# they are also driven by the CLI in __main__. pytest treats those parameters as fixture
# requests, so without these definitions every test here errored at setup with
# "fixture 'ann_model' not found" — the entire file was uncollectable under pytest.
#
# Set STAC_TEST_MODEL to a local path to run offline; the fixtures skip (rather than
# fail) when the model cannot be loaded.

def _pytest_model_name():
    return os.environ.get("STAC_TEST_MODEL", "distilgpt2")


@pytest.fixture(scope="module")
def args():
    return parse_args([
        "--model_name", _pytest_model_name(),
        "--timesteps", "2",
        "--test_turns", "2",
        "--output_dir", os.path.join(tempfile.gettempdir(), "stac_pytest_output"),
    ])


def _load_base_model(args):
    try:
        return AutoModelForCausalLM.from_pretrained(args.model_name)
    except Exception as e:  # offline, missing model, etc.
        pytest.skip(f"Could not load model {args.model_name!r}: {e}")


@pytest.fixture(scope="module")
def tokenizer(args):
    try:
        tok = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        pytest.skip(f"Could not load tokenizer {args.model_name!r}: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def ann_model(args):
    model = _load_base_model(args)
    model.eval()
    return model


@pytest.fixture(scope="module")
def snn_model(args):
    # A separate instance: conversion rewrites modules in place, so the SNN must not
    # share state with the ANN reference model.
    base = _load_base_model(args)
    base.T = args.timesteps
    converted = simplified_conversion(base, args.timesteps, skip_gelu_replacement=True)
    converted.eval()
    return converted


@pytest.fixture(scope="module")
def model(snn_model):
    """Alias used by the tests that only need the converted model."""
    return snn_model


def _get_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 1:
        return outputs[0]
    return outputs


def test_fidelity_parity(ann_model, snn_model, tokenizer, args):
    """
    Compare ANN vs SNN logits on fixed prompts and report:
    - max_abs_diff of last-position logits
    - top-1 next-token agreement rate
    """
    logger.info("Running: test_fidelity_parity")
    device = args.device

    prompts = [
        "The capital of France is",
        "2+2=",
        "In one sentence, explain gravity:",
        "Alice went to the market and bought",
    ]

    ann_model.eval()
    snn_model.eval()

    max_abs_diffs = []
    top1_matches = 0
    total = 0

    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            ann_out = ann_model(**inputs)
            # Ensure parity tests are not affected by KV-cache carryover in wrappers.
            try:
                snn_out = snn_model(inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)
            except TypeError:
                snn_out = snn_model(inputs.input_ids, attention_mask=inputs.attention_mask)

        ann_logits = _get_logits(ann_out)
        snn_logits = _get_logits(snn_out)

        ann_last = ann_logits[0, -1, :].float().cpu()
        snn_last = snn_logits[0, -1, :].float().cpu()

        diff = (ann_last - snn_last).abs()
        max_abs = float(diff.max().item())
        max_abs_diffs.append(max_abs)

        ann_tok = int(torch.argmax(ann_last).item())
        snn_tok = int(torch.argmax(snn_last).item())
        top1_matches += int(ann_tok == snn_tok)
        total += 1

        logger.info(f"Parity prompt: {p}")
        logger.info(f"  max_abs_diff={max_abs:.6f}")
        logger.info(f"  ann_top1={ann_tok} ({_safe_console_text(tokenizer.decode([ann_tok]))})")
        logger.info(f"  snn_top1={snn_tok} ({_safe_console_text(tokenizer.decode([snn_tok]))})")

    max_of_max = max(max_abs_diffs) if max_abs_diffs else float("inf")
    top1_rate = (top1_matches / total) if total else 0.0

    logger.info(f"Fidelity summary: prompts={total}, top1_match_rate={top1_rate:.2%}, max_abs_diff_max={max_of_max:.6f}")

    # Parity gate: default to "either" because top-1 agreement is often a more meaningful operational metric
    # than raw logit deltas (which can be large even when argmax matches).
    top1_min = float(getattr(args, "fidelity_top1_min", 0.50))
    gate = str(getattr(args, "fidelity_gate", "either")).lower()
    if gate == "logits":
        passed = (max_of_max <= float(args.parity_tolerance))
    elif gate == "top1":
        passed = (top1_rate >= top1_min)
    else:
        passed = (top1_rate >= top1_min) or (max_of_max <= float(args.parity_tolerance))
    if passed:
        logger.info("PASS: test_fidelity_parity (within tolerance)")
    else:
        logger.error("FAIL: test_fidelity_parity (exceeds tolerance)")
    # Assert as well as return: pytest ignores a returned False, so returning the result
    # alone made this test pass under pytest no matter what it measured.
    assert passed, "ANN<->SNN fidelity parity failed"
    return passed


def test_multi_turn_parity(ann_model, snn_model, tokenizer, args):
    """
    Deterministic greedy next-token trajectory comparison on a short multi-turn script.
    We compare the next-token ID at each step for N steps.
    """
    logger.info("Running: test_multi_turn_parity")
    device = args.device
    steps = int(args.parity_steps)

    history = "User: My name is Alice.\nAssistant: Hello Alice.\nUser: What is my name?\nAssistant:"
    input_ids = tokenizer(history, return_tensors="pt").input_ids.to(device)

    ann_model.eval()
    snn_model.eval()
    # Avoid accidental cache carryover inside TemporalSpikeProcessor across repeated full-context forwards.
    if hasattr(snn_model, "reset_cache"):
        snn_model.reset_cache()

    matches = 0
    compared = 0

    cur_ids_ann = input_ids.clone()
    cur_ids_snn = input_ids.clone()

    for _i in range(steps):
        attn_ann = torch.ones_like(cur_ids_ann, device=device)
        attn_snn = torch.ones_like(cur_ids_snn, device=device)

        with torch.no_grad():
            ann_logits = _get_logits(ann_model(cur_ids_ann, attention_mask=attn_ann))
            try:
                snn_logits = _get_logits(snn_model(cur_ids_snn, attention_mask=attn_snn, use_cache=False))
            except TypeError:
                snn_logits = _get_logits(snn_model(cur_ids_snn, attention_mask=attn_snn))

        ann_next = int(torch.argmax(ann_logits[0, -1, :]).item())
        snn_next = int(torch.argmax(snn_logits[0, -1, :]).item())

        matches += int(ann_next == snn_next)
        compared += 1

        cur_ids_ann = torch.cat([cur_ids_ann, torch.tensor([[ann_next]], device=device)], dim=1)
        cur_ids_snn = torch.cat([cur_ids_snn, torch.tensor([[snn_next]], device=device)], dim=1)

    match_rate = (matches / compared) if compared else 0.0
    logger.info(f"Multi-turn parity summary: steps={compared}, next_token_match_rate={match_rate:.2%}")

    passed = match_rate >= 0.10
    if passed:
        logger.info("PASS: test_multi_turn_parity (minimum agreement met)")
    else:
        logger.error("FAIL: test_multi_turn_parity (insufficient agreement)")
    assert passed, "ANN<->SNN multi-turn parity failed"
    return passed


def test_tsp_state_retention(model, tokenizer, args):
    """Deterministic test that TSP can preserve context across turns in incremental mode."""
    logger.info("Running: test_tsp_state_retention")

    # Ensure we have the wrapper
    if not isinstance(model, TemporalSpikeProcessor):
        model = TemporalSpikeProcessor(model, T=args.timesteps, max_context_length=min(args.max_context_length, 256))

    model.reset_cache()

    t1 = "User: Hello.\nAssistant:"
    t2 = "\nUser: What is 2+2?\nAssistant:"

    ids1 = tokenizer(t1, return_tensors="pt").input_ids.to(args.device)
    ids2 = tokenizer(t2, return_tensors="pt").input_ids.to(args.device)

    # Feed turn 1 incrementally
    _ = model(ids1, incremental=True)
    len1 = model.get_cached_input_length() if hasattr(model, "get_cached_input_length") else 0
    if len1 <= 0:
        logger.error("FAIL: TSP did not create token cache in incremental mode after first turn.")
        pytest.fail("TSP did not create a token cache in incremental mode after the first turn")
        return False

    # Feed turn 2 incrementally
    _ = model(ids2, incremental=True)
    len2 = model.get_cached_input_length() if hasattr(model, "get_cached_input_length") else 0
    if len2 <= len1:
        logger.error(f"FAIL: TSP cache did not grow across turns (len1={len1}, len2={len2}).")
        pytest.fail(f"TSP cache did not grow across turns (len1={len1}, len2={len2})")
        return False

    # Position ids should match cached length
    pos = model.get_position_ids() if hasattr(model, "get_position_ids") else None
    if pos is None or pos.numel() == 0:
        logger.error("FAIL: TSP did not expose position ids.")
        pytest.fail("TSP did not expose position ids")
        return False
    if int(pos.max().item()) != (len2 - 1):
        logger.error(f"FAIL: Position IDs max does not match cached length-1 (pos_max={int(pos.max().item())}, expected={len2-1}).")
        pytest.fail(f"Position IDs max {int(pos.max().item())} != cached length-1 {len2-1}")
        return False

    # Reset should clear
    model.reset_cache()
    len0 = model.get_cached_input_length() if hasattr(model, "get_cached_input_length") else 0
    if len0 != 0:
        logger.error(f"FAIL: TSP reset_cache did not clear token cache (len0={len0}).")
        pytest.fail(f"TSP reset_cache did not clear the token cache (len0={len0})")
        return False

    logger.info("PASS: test_tsp_state_retention")
    return True

def simulate_conversation(model, tokenizer, turns=3, device="cpu", max_context_length=512):
    """Simulate a conversation with the model and verify state handling."""
    logger.info(f"Testing {turns} conversation turns")
    
    # Set up a test conversation
    conversation = [
        "Hello, how are you today?",
        "What's your favorite color?",
        "Tell me more about that color.",
        "Do you like other colors too?",
        "Thank you for chatting with me!"
    ]
    
    # Use only the first N turns based on parameter
    test_prompts = conversation[:turns]
    
    # Initialize conversation history
    history = []
    
    # Initialize the model's state
    if hasattr(model, 'reset_cache'):
        model.reset_cache()
    
    # Keep track of tokens for attention mask
    conv_tokens = None
    
    # Process each turn
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\nTurn {i+1}: User: {prompt}")
        
        # Format the prompt with history
        if not history:
            formatted_prompt = f"User: {prompt}\nAssistant: "
            # Tokenize the full prompt
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            conv_tokens = inputs.input_ids
        else:
            # Add to existing conversation
            formatted_prompt = f"\nUser: {prompt}\nAssistant: "
            # Tokenize just the new input
            new_tokens = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            # Append to conversation history
            conv_tokens = torch.cat([conv_tokens, new_tokens.input_ids], dim=1)
        
        # Handle position IDs for longer sequences
        # Clamp the size to prevent position embedding index errors
        if conv_tokens.size(1) > max_context_length:
            logger.warning(f"Sequence length {conv_tokens.size(1)} exceeds max length {max_context_length}. Truncating.")
            conv_tokens = conv_tokens[:, -max_context_length:]
        
        # Generate a response - for testing, limit to 30 tokens per turn
        max_new_tokens = 30
        response_tokens = []
        
        # Set model to evaluation mode
        model.eval()
        
        # Generate output tokens
        with torch.no_grad():
            # Create a proper padding-compatible attention mask
            # All 1s indicates "attend to all tokens"
            attention_mask = torch.ones((1, conv_tokens.size(1)), device=device)
            
            for j in range(max_new_tokens):
                # Forward pass with the conversation history
                try:
                    # Reset cache before each forward pass to avoid state contamination
                    if hasattr(model, 'reset_cache'):
                        model.reset_cache()
                    # Pass attention mask to handle the context properly (use_cache=False for clean computation)
                    outputs = model(
                        conv_tokens,
                        attention_mask=attention_mask,
                        use_cache=False
                    )
                    
                    # Get next token with improved sampling to avoid repetition
                    next_token_logits = outputs[0, -1, :].clone()
                    
                    # Blacklist problematic tokens that cause loops
                    blacklist_tokens = [11, 12, 198]  # comma, dash, newline
                    for token_id in blacklist_tokens:
                        next_token_logits[token_id] = -float('inf')
                    
                    # Strong repetition penalty
                    if len(response_tokens) >= 2:
                        recent_tokens = response_tokens[-2:]
                        for token_id in recent_tokens:
                            next_token_logits[token_id] -= 10.0  # Strong penalty
                    
                    # Apply temperature
                    temperature = 1.2  # Higher temperature for more diversity
                    next_token_logits = next_token_logits / temperature
                    
                    # Use top-k sampling
                    top_k = 100
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1).item()
                    next_token_id = top_k_indices[next_token_idx].item()
                    
                    # If EOS token, stop generation
                    if next_token_id == tokenizer.eos_token_id:
                        break
                    
                    # Store token
                    response_tokens.append(next_token_id)
                    
                    # Update conversation tokens
                    conv_tokens = torch.cat([
                        conv_tokens, 
                        torch.tensor([[next_token_id]], device=device)
                    ], dim=1)
                    
                    # Keep length within max_context_length
                    if conv_tokens.size(1) > max_context_length:
                        conv_tokens = conv_tokens[:, -max_context_length:]
                    
                    # Update attention mask
                    attention_mask = torch.ones((1, conv_tokens.size(1)), device=device)
                except Exception as e:
                    logger.error(f"Error during generation step {j}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        
        # Decode the response
        response_text = tokenizer.decode(response_tokens)
        logger.info(f"Turn {i+1} Assistant: {_safe_console_text(response_text)}")
        
        # Add to history for next turn
        history.append(f"User: {prompt}")
        history.append(f"Assistant: {response_text}")
        
        # Check that the model's KV cache and state is maintained
        if i > 0:
            logger.info(f"  - Verified turn {i+1} processed with history from previous turns")
        
        # Verify position IDs
        if hasattr(model, 'get_position_ids'):
            position_ids = model.get_position_ids()
            logger.info(f"  - Position IDs: {position_ids}")
            # Verify implementation
            assert torch.all(position_ids >= 0).item(), "Position IDs should be non-negative"
            # Additional check matching requirement
            assert position_ids.max().item() >= 0, "Position IDs should be properly managed"
    
    # Test passed if it reaches here without errors
    logger.info("\nConversation test completed successfully.")
    return True

def test_position_id_boundaries(model, tokenizer, args):
    """Verify position IDs stay within model's max_position_embeddings"""
    logger.info("Running: test_position_id_boundaries")
    
    device = args.device if hasattr(args, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')

    if not hasattr(model, 'config') or not hasattr(model.config, 'max_position_embeddings'):
        logger.warning("Model or model.config lacks 'max_position_embeddings'. Using fallback or skipping some checks.")
        max_pos = model.max_context_length if hasattr(model, 'max_context_length') else 2048 # Fallback to max_context_length
    else:
        max_pos = model.config.max_position_embeddings
    
    logger.info(f"Effective max_pos for test: {max_pos}")

    # Test sequence at max length
    logger.info(f"Testing with sequence at effective max_pos: {max_pos}")
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_pos), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        # Check if model provides position IDs
        if hasattr(model, 'get_position_ids'):
            position_ids = model.get_position_ids()
            # Verify position IDs are within bounds
            assert position_ids.max().item() < max_pos, f"Position IDs exceed max_position_embeddings: {position_ids.max().item()} >= {max_pos}"
            assert position_ids.min().item() >= 0, f"Position IDs contain negative values: {position_ids.min().item()}"
            logger.info(f"Position IDs verified within bounds: min={position_ids.min().item()}, max={position_ids.max().item()}, limit={max_pos}")
        
        # Validate output shape matches input
        assert outputs.logits.shape[0] == input_ids.shape[0], f"Batch size mismatch: {outputs.logits.shape[0]} != {input_ids.shape[0]}"
        assert outputs.logits.shape[1] == input_ids.shape[1], f"Sequence length mismatch: {outputs.logits.shape[1]} != {input_ids.shape[1]}"
        logger.info("Forward pass at effective max_pos completed with correct output shapes.")
    except Exception as e:
        logger.error(f"Model forward pass failed at effective max_pos ({max_pos}): {e}")
        pytest.fail(f"Model forward pass failed at effective max_pos ({max_pos}): {e}")
        return False

    # Test overflow handling: sequence longer than max_pos_embeddings
    # TemporalSpikeProcessor clamps position_ids and truncates input_ids based on max_context_length.
    if hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
        test_overflow_len = model.config.max_position_embeddings + 10
        logger.info(f"Testing with sequence ({test_overflow_len}) longer than actual max_position_embeddings ({model.config.max_position_embeddings})")
        long_input_ids = torch.randint(0, tokenizer.vocab_size, (1, test_overflow_len), device=device)
        long_attention_mask = torch.ones_like(long_input_ids)
        
        try:
            with torch.no_grad():
                outputs = model(long_input_ids, attention_mask=long_attention_mask)
            
            # Verify position IDs clamping behavior
            if hasattr(model, 'get_position_ids'):
                position_ids = model.get_position_ids()
                # Verify position IDs are clamped within bounds
                assert position_ids.max().item() < max_pos, f"Position IDs not clamped: {position_ids.max().item()} >= {max_pos}"
                logger.info(f"Position IDs correctly clamped: max={position_ids.max().item()}, limit={max_pos}")
            
            # The model truncates its *context* internally, but the logits it returns
            # must still line up with the input the caller passed — the same contract
            # asserted for the at-max_pos case above. (This previously expected the
            # truncated length, contradicting that assertion and the wrapper's own
            # documented behaviour.)
            assert outputs.logits.shape[1] == test_overflow_len, \
                f"Output sequence length incorrect: {outputs.logits.shape[1]} != {test_overflow_len}"
            effective_context = min(
                test_overflow_len,
                getattr(model, 'max_context_length', test_overflow_len),
                max_pos,
            )
            logger.info(
                f"Model handled input of length {test_overflow_len} correctly "
                f"(attended over the last {effective_context} positions)."
            )
        except Exception as e:
            logger.error(f"Model forward pass failed for long_input (length {test_overflow_len}): {e}")
            pytest.fail(f"Model failed on input longer than max_position_embeddings: {e}")
            return False
    else:
        logger.info("Skipping explicit position embedding overflow test as model.config.max_position_embeddings not found.")

    logger.info("PASS: test_position_id_boundaries (adapted for SNN wrapper behavior).")
    return True

def test_padded_batch_is_finite(model, tokenizer, args):
    """
    A batch containing padding must not produce NaN/inf logits.

    Regression test: SpikeAttention treated HuggingFace's *additive* 4D mask (0 = attend,
    -3.4e38 = masked) as a 0/1 keep-mask and re-inverted it, turning masked positions into
    -inf. Entire softmax rows became -inf and every logit came back NaN — so any batched
    inference with padding silently produced garbage.
    """
    logger.info("Running: test_padded_batch_is_finite")
    device = args.device

    # The model fixture is shared across tests; start from a clean cache.
    if hasattr(model, 'reset_cache'):
        model.reset_cache()

    vocab = int(getattr(model.config, 'vocab_size', tokenizer.vocab_size))
    torch.manual_seed(0)
    input_ids = torch.randint(0, min(vocab, tokenizer.vocab_size), (2, 10), device=device)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :4] = 0  # second sequence is padded

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    assert torch.isfinite(logits).all(), (
        "Padded batch produced non-finite logits: "
        f"{int(torch.isnan(logits).sum())} NaN, {int(torch.isinf(logits).sum())} inf"
    )
    logger.info("PASS: test_padded_batch_is_finite")
    return True


def test_attention_mask_continuity(model, tokenizer, args):
    """Verify attention mask grows correctly across turns and properly handles edge cases."""
    logger.info("Running: test_attention_mask_continuity")
    device = args.device if hasattr(args, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')

    if hasattr(model, 'reset_cache'): 
        model.reset_cache()
        logger.info("Model cache reset successfully")
    
    # Initial input
    text1 = "Hello world."
    input_ids_turn1 = tokenizer(text1, return_tensors="pt").to(device).input_ids
    attention_mask_turn1 = torch.ones_like(input_ids_turn1)
    
    logger.info(f"Turn 1: Input length {input_ids_turn1.shape[1]}")
    try:
        with torch.no_grad():
            outputs_turn1 = model(input_ids_turn1, attention_mask=attention_mask_turn1)
        
        # Validate initial output shape
        assert hasattr(outputs_turn1, 'logits') or isinstance(outputs_turn1, torch.Tensor), "Model output does not have logits attribute or is not a tensor"
        logits_turn1 = outputs_turn1.logits if hasattr(outputs_turn1, 'logits') else outputs_turn1
        assert logits_turn1.shape[1] == input_ids_turn1.shape[1], f"Output sequence length mismatch: {logits_turn1.shape[1]} != {input_ids_turn1.shape[1]}"
        logger.info(f"Turn 1 output validated: shape {logits_turn1.shape}")
        
        # Simulate generating one token
        next_token_id_turn1 = torch.argmax(logits_turn1[0, -1, :]).unsqueeze(0).unsqueeze(0)

        # Input for turn 2 (previous + new question + generated token from turn1)
        text2 = " How are you?"
        input_ids_text2 = tokenizer(text2, return_tensors="pt").to(device).input_ids
        
        input_ids_turn2 = torch.cat([input_ids_turn1, next_token_id_turn1, input_ids_text2], dim=1)
        # Create attention mask for this combined input
        attention_mask_turn2 = torch.ones_like(input_ids_turn2)

        # Save the original length for validation
        original_length_turn2 = input_ids_turn2.shape[1]

        # Check if we need to truncate due to model constraints
        model_max_len = getattr(model, 'max_context_length', 2048)
        if input_ids_turn2.shape[1] > model_max_len:
            logger.info(f"Truncating turn 2 input from {input_ids_turn2.shape[1]} to {model_max_len}")
            input_ids_turn2 = input_ids_turn2[:, -model_max_len:]
            attention_mask_turn2 = attention_mask_turn2[:, -model_max_len:]
            
            # Validate truncation was done correctly
            assert input_ids_turn2.shape[1] == model_max_len, f"Truncation failed: {input_ids_turn2.shape[1]} != {model_max_len}"
            assert attention_mask_turn2.shape[1] == model_max_len, f"Attention mask truncation failed: {attention_mask_turn2.shape[1]} != {model_max_len}"
            assert torch.all(attention_mask_turn2 == 1), "Truncated attention mask values aren't all 1s"

        logger.info(f"Turn 2: Input length {input_ids_turn2.shape[1]}")
        with torch.no_grad():
            outputs_turn2 = model(input_ids_turn2, attention_mask=attention_mask_turn2)

        # Validate turn 2 output
        logits_turn2 = outputs_turn2.logits if hasattr(outputs_turn2, 'logits') else outputs_turn2
        assert logits_turn2.shape[1] == input_ids_turn2.shape[1], f"Turn 2 output sequence length mismatch: {logits_turn2.shape[1]} != {input_ids_turn2.shape[1]}"
        logger.info(f"Turn 2 output validated: shape {logits_turn2.shape}")

        # Test KV cache handling if the model supports it
        if hasattr(model, 'kv_cache') and model.kv_cache is not None and len(model.kv_cache) > 0:
            # Check cache shape after second pass
            cache_len_after_turn2 = model.kv_cache[0][0].shape[2]
            # This should match the input length for turn 2 (or be capped at max_context_length)
            expected_cache_len = min(input_ids_turn2.shape[1], model_max_len)
            assert cache_len_after_turn2 == expected_cache_len, \
                f"KV cache length {cache_len_after_turn2} != expected {expected_cache_len}"
            logger.info(f"KV cache correctly maintained: length {cache_len_after_turn2}")
        
        # Test step-by-step generation with growing masks
        if hasattr(model, 'reset_cache'): 
            model.reset_cache()
        current_full_input_ids = tokenizer("Step-by-step test:", return_tensors="pt").to(device).input_ids
        current_mask = torch.ones_like(current_full_input_ids)

        for step in range(3):  # Simulate generating 3 tokens
            prev_ids_len = current_full_input_ids.shape[1]
            prev_mask_len = current_mask.shape[1]
            
            with torch.no_grad():
                outputs = model(current_full_input_ids, attention_mask=current_mask)
        
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            next_token = torch.argmax(logits[0,-1,:]).unsqueeze(0).unsqueeze(0)
            
            # Add new token to input
            current_full_input_ids = torch.cat([current_full_input_ids, next_token], dim=1)
            # Extend attention mask with a 1 for the new token
            current_mask = torch.cat([current_mask, torch.ones((1,1), device=device, dtype=torch.long)], dim=1)

            # Validate mask and input shapes are consistent
            assert current_full_input_ids.shape[1] == prev_ids_len + 1, \
                f"Step {step+1}: Input IDs length {current_full_input_ids.shape[1]} != expected {prev_ids_len + 1}"
            assert current_mask.shape[1] == prev_mask_len + 1, \
                f"Step {step+1}: Mask length {current_mask.shape[1]} != expected {prev_mask_len + 1}"
            assert current_mask.shape == current_full_input_ids.shape, \
                f"Step {step+1}: Mask shape {current_mask.shape} != input shape {current_full_input_ids.shape}"
            assert torch.all(current_mask[0, -1] == 1).item(), \
                f"Step {step+1}: New token mask in constructed mask not set to 1"
    
            logger.info("Mask growth verified for step-by-step generation simulation.")

        # Test edge case: Zero-length masks
        try:
            # Create a 0-length mask to ensure the model handles this gracefully
            zero_ids = torch.zeros((1, 0), dtype=torch.long, device=device)
            zero_mask = torch.zeros((1, 0), dtype=torch.long, device=device)
            
            # This should raise an error as expected, so we'll catch it and verify the error message
            # is related to the empty tensor and not something else
            with torch.no_grad():
                model(zero_ids, attention_mask=zero_mask)
            
            # If we got here, no error was raised - this might be fine if the model handles empty inputs
            logger.info("Model handled zero-length mask without error (acceptable behavior)")
        except Exception as e:
            # We expect an error here, but it should be the right kind of error 
            # (related to empty tensor, not a generic failure)
            error_msg = str(e).lower()
            expected_errors = ["empty", "zero", "shape", "dimension", "length"]
            if any(err in error_msg for err in expected_errors):
                logger.info(f"Model correctly raised appropriate error for zero-length mask: {e}")
            else:
                logger.error(f"Model raised unexpected error for zero-length mask: {e}")
                pytest.fail(f"Unexpected error for zero-length mask: {e}")
                return False

    except Exception as e:
        logger.error(f"Error during attention mask continuity test: {e}")
        pytest.fail(f"Error during attention mask continuity test: {e}")
        return False
            
    logger.info("PASS: test_attention_mask_continuity")
    return True

def test_multi_turn_coherence(model, tokenizer, args):
    """Validate context retention across conversation turns with specific coherence tests."""
    logger.info("Running: test_multi_turn_coherence")
    device = args.device if hasattr(args, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
    max_new_tokens_per_turn = args.max_new_tokens_per_turn if hasattr(args, 'max_new_tokens_per_turn') else 20 # Default

    # Reset model state
    if hasattr(model, 'reset_cache'): 
        model.reset_cache()
        logger.info("Model cache reset successfully")
    
    # Test scenarios with specific context information that should be maintained
    coherence_tests = [
        # Test 1: Name recall
        [
            ("My name is Alice Smith from New York.", ["alice", "smith", "new york"]), 
            ("What is my name?", ["alice", "smith"]),
            ("Where am I from?", ["new york"])
        ],
        
        # Test 2: Numerical information retention
        [
            ("I have 3 dogs and 2 cats.", ["3", "dogs", "2", "cats"]),
            ("How many pets do I have?", ["5", "pets", "animals"]),
            ("How many dogs do I have?", ["3", "dogs"]),
            ("How many cats do I have?", ["2", "cats"])
        ],
        
        # Test 3: Contextual fact retention
        [
            ("The capital of France is Paris, which is known for the Eiffel Tower.", ["paris", "france", "eiffel"]),
            ("What is the capital of France?", ["paris"]),
            ("What is Paris known for?", ["eiffel", "tower"])
        ]
    ]
    
    all_tests_passed = True
    all_contexts = []
    
    for test_idx, test_scenario in enumerate(coherence_tests):
        logger.info(f"\n=== Coherence Test {test_idx+1} ===")
        
        # Reset for each test scenario
        if hasattr(model, 'reset_cache'): model.reset_cache()
        context_history_for_input_str = "" # String to build up the full conversation history
        accumulated_input_ids = None
        
        for turn_idx, (question_text, expected_keywords) in enumerate(test_scenario):
            logger.info(f"\nTurn {turn_idx+1}: \"{question_text}\"")
            logger.info(f"Expected keywords: {expected_keywords}")
            
            # Format as user/assistant conversation
            new_turn_text = f"\nUser: {question_text}\nAssistant: " if turn_idx > 0 else f"User: {question_text}\nAssistant: "
            
            # For first turn, just use the question
            if turn_idx == 0:
                context_history_for_input_str = new_turn_text
                current_input_ids = tokenizer(context_history_for_input_str, return_tensors="pt").to(device).input_ids
                accumulated_input_ids = current_input_ids
            else:
                # For subsequent turns, use the accumulated history + new question
                new_turn_ids = tokenizer(new_turn_text, return_tensors="pt").to(device).input_ids
                accumulated_input_ids = torch.cat([accumulated_input_ids, new_turn_ids], dim=1)
            
            # Handle context length constraints
            model_max_len = model.max_context_length if hasattr(model, 'max_context_length') else 2048
            if accumulated_input_ids.shape[1] > model_max_len:
                logger.warning(f"Input length {accumulated_input_ids.shape[1]} exceeds model_max_len {model_max_len}. Truncating.")
                accumulated_input_ids = accumulated_input_ids[:, -model_max_len:]
                
                # Validate truncation was done correctly
                assert accumulated_input_ids.shape[1] <= model_max_len, \
                    f"Truncation failed: {accumulated_input_ids.shape[1]} > {model_max_len}"
            
            # Generate response with validated input
            logger.info(f"Feeding context of length {accumulated_input_ids.shape[1]} tokens to model")
            attention_mask = torch.ones_like(accumulated_input_ids)
            
            # Generate response tokens
            generated_ids_list = []
            model.eval()

            with torch.no_grad():
                for step in range(max_new_tokens_per_turn):
                    # Reset cache before each forward pass to avoid state contamination
                    if hasattr(model, 'reset_cache'):
                        model.reset_cache()
                    # Forward pass with full history (use_cache=False for clean computation)
                    outputs = model(accumulated_input_ids, attention_mask=attention_mask, use_cache=False)
                
                    # Get next token prediction
                    next_token_logits = outputs.logits[0, -1, :] if hasattr(outputs, 'logits') else outputs[0, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                    # Stop if EOS token
                    if next_token_id == tokenizer.eos_token_id:
                        break
                
                    # Add to generated tokens
                    generated_ids_list.append(next_token_id)
                
                    # Add to accumulated input for next step
                    next_token_tensor = torch.tensor([[next_token_id]], device=device)
                    accumulated_input_ids = torch.cat([accumulated_input_ids, next_token_tensor], dim=1)
                    attention_mask = torch.ones_like(accumulated_input_ids)
                
                    # Check if we're approaching model max length
                    if accumulated_input_ids.shape[1] >= model_max_len - 5:
                        logger.warning(f"Approaching max context length. Stopping generation at {step+1} tokens.")
                        break
            
            # Convert generated tokens to text
            generated_text = tokenizer.decode(generated_ids_list)
            context_history_for_input_str += generated_text
            
            logger.info(f"Generated: \"{generated_text}\"")
            
            # Check for expected keywords in the response
            keywords_found = []
            keywords_missing = []
            
            for keyword in expected_keywords:
                if keyword.lower() in generated_text.lower():
                    keywords_found.append(keyword)
                else:
                    keywords_missing.append(keyword)
            
            # Determine if enough keywords were found (at least 1, or 50% of expected)
            keywords_threshold = max(1, len(expected_keywords) // 2)
            keywords_test_passed = len(keywords_found) >= keywords_threshold
            
            if keywords_test_passed:
                logger.info(f"PASS: Found {len(keywords_found)}/{len(expected_keywords)} expected keywords: {keywords_found}")
                if keywords_missing:
                    logger.info(f"   Missing keywords: {keywords_missing}")
            else:
                logger.error(f"FAIL: Only found {len(keywords_found)}/{len(expected_keywords)} expected keywords: {keywords_found}")
                logger.error(f"   Missing critical keywords: {keywords_missing}")
                all_tests_passed = False
            
            # Store context for final verification
            all_contexts.append({
                'test_idx': test_idx,
                'turn_idx': turn_idx,
                'question': question_text,
                'response': generated_text,
                'expected_keywords': expected_keywords,
                'found_keywords': keywords_found,
                'missing_keywords': keywords_missing,
                'passed': keywords_test_passed
            })
    
    # Final summary
    logger.info("\n=== Multi-turn Coherence Test Summary ===")
    tests_passed = 0
    tests_failed = 0
    
    for ctx in all_contexts:
        if ctx['passed']:
            tests_passed += 1
        else:
            tests_failed += 1
            logger.error(f"Failed: Test {ctx['test_idx']+1}, Turn {ctx['turn_idx']+1}")
            logger.error(f"  Question: \"{ctx['question']}\"")
            logger.error(f"  Response: \"{ctx['response']}\"")
            logger.error(f"  Missing keywords: {ctx['missing_keywords']}")
    
    pass_rate = (tests_passed / (tests_passed + tests_failed)) * 100 if (tests_passed + tests_failed) > 0 else 0
    logger.info(f"Tests passed: {tests_passed}/{tests_passed + tests_failed} ({pass_rate:.1f}%)")
    
    # Overall test passes if a majority of keyword tests pass (80% or higher)
    overall_pass_threshold = 0.8
    overall_pass = pass_rate >= (overall_pass_threshold * 100)
    
    if overall_pass:
        logger.info(f"PASS: test_multi_turn_coherence with {pass_rate:.1f}% success rate")
    else:
        logger.error(f"FAIL: test_multi_turn_coherence with only {pass_rate:.1f}% success rate (threshold: {overall_pass_threshold * 100:.1f}%)")

    assert overall_pass, (
        f"multi-turn coherence success rate {pass_rate:.1f}% is below the "
        f"{overall_pass_threshold * 100:.1f}% threshold"
    )
    return overall_pass

def test_energy_consumption(model, tokenizer, args):
    """Validate spike-based efficiency improvements using torch.profiler for both CPU/CUDA time and memory usage."""
    logger.info("Running: test_energy_consumption")
    device = args.device if hasattr(args, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')

    ann_model_name = args.model_name # Assuming SNN is based on this ANN model
    try:
        logger.info(f"Loading ANN base model: {ann_model_name} for comparison.")
        ann_model = AutoModelForCausalLM.from_pretrained(ann_model_name).to(device)
        ann_model.eval()
    except Exception as e:
        logger.error(f"Failed to load ANN model {ann_model_name} for energy comparison: {e}")
        pytest.fail(f"Failed to load ANN model {ann_model_name}: {e}")
        return False

    snn_model = model # This is already loaded and passed in
    snn_model.eval()

    # Profiler traces are written here too; create it up front rather than failing
    # mid-measurement.
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Prepare multiple inputs with different sequence lengths for thorough testing
    test_lengths = [32, 64, 128]
    test_inputs = []
    for length in test_lengths:
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, length), device=device)
        attention_mask = torch.ones_like(input_ids)
        test_inputs.append((input_ids, attention_mask))

    # Warmup runs to eliminate startup overhead
    logger.info("Performing warmup runs...")
    for _ in range(5):
        for input_ids, attention_mask in test_inputs:
            with torch.no_grad():
                _ = ann_model(input_ids, attention_mask=attention_mask)
                _ = snn_model(input_ids, attention_mask=attention_mask)
    
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == 'cuda' and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        logger.info("CUDA profiling enabled")

    # Track metrics for all test sequences
    ann_metrics = {length: {} for length in test_lengths}
    snn_metrics = {length: {} for length in test_lengths}

    # Profile ANN model
    logger.info("Profiling ANN model...")
    for i, (input_ids, attention_mask) in enumerate(test_inputs):
        length = test_lengths[i]
        logger.info(f"Profiling ANN with sequence length {length}...")
        
        # Track memory before and after
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        
        try:
            with torch.profiler.profile(
                activities=activities, 
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as ann_prof:
                with torch.no_grad():
                    for _ in range(PROFILE_REPEATS):
                        ann_model(input_ids, attention_mask=attention_mask)

            # Process profiler results (averaged over PROFILE_REPEATS forwards: a single
            # cold, profiled call varied by ~10x run to run and made this test flaky).
            ann_total_cpu_time_us = sum(evt.cpu_time_total for evt in ann_prof.key_averages()) / PROFILE_REPEATS
            ann_total_cuda_time_us = sum(_accelerator_time_total(evt) for evt in ann_prof.key_averages()) / PROFILE_REPEATS
            ann_total_time_us = ann_total_cpu_time_us + ann_total_cuda_time_us
            
            # Track memory usage if on CUDA
            if device == 'cuda' and torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2) - initial_memory  # MB
                ann_metrics[length]['memory_mb'] = peak_memory
                logger.info(f"ANN memory usage for length {length}: {peak_memory:.2f} MB")
            
            ann_metrics[length]['cpu_time_ms'] = ann_total_cpu_time_us / 1000
            ann_metrics[length]['cuda_time_ms'] = ann_total_cuda_time_us / 1000
            ann_metrics[length]['total_time_ms'] = ann_total_time_us / 1000
            
            logger.info(f"ANN time for length {length}: {ann_total_time_us / 1000:.2f} ms (CPU: {ann_total_cpu_time_us / 1000:.2f} ms, CUDA: {ann_total_cuda_time_us / 1000:.2f} ms)")
            
            # Save profile trace for analysis
            if args.output_dir:
                trace_path = os.path.join(args.output_dir, f"ann_profile_length_{length}.json")
                ann_prof.export_chrome_trace(trace_path)
                logger.info(f"Saved ANN profile trace to {trace_path}")

        except Exception as e:
            logger.error(f"Error profiling ANN model at sequence length {length}: {e}")
            pytest.fail(f"Error profiling ANN model: {e}")
            return False

    # Profile SNN model
    logger.info("Profiling SNN model...")
    for i, (input_ids, attention_mask) in enumerate(test_inputs):
        length = test_lengths[i]
        logger.info(f"Profiling SNN with sequence length {length}...")
        
        # Track memory before and after
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        
        try:
            with torch.profiler.profile(
                activities=activities, 
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as snn_prof:
                with torch.no_grad():
                    for _ in range(PROFILE_REPEATS):
                        snn_model(input_ids, attention_mask=attention_mask)

            # Process profiler results
            snn_total_cpu_time_us = sum(evt.cpu_time_total for evt in snn_prof.key_averages()) / PROFILE_REPEATS
            snn_total_cuda_time_us = sum(_accelerator_time_total(evt) for evt in snn_prof.key_averages()) / PROFILE_REPEATS
            snn_total_time_us = snn_total_cpu_time_us + snn_total_cuda_time_us
            
            # Track memory usage if on CUDA
            if device == 'cuda' and torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2) - initial_memory  # MB
                snn_metrics[length]['memory_mb'] = peak_memory
                logger.info(f"SNN memory usage for length {length}: {peak_memory:.2f} MB")
            
            snn_metrics[length]['cpu_time_ms'] = snn_total_cpu_time_us / 1000
            snn_metrics[length]['cuda_time_ms'] = snn_total_cuda_time_us / 1000
            snn_metrics[length]['total_time_ms'] = snn_total_time_us / 1000
            
            logger.info(f"SNN time for length {length}: {snn_total_time_us / 1000:.2f} ms (CPU: {snn_total_cpu_time_us / 1000:.2f} ms, CUDA: {snn_total_cuda_time_us / 1000:.2f} ms)")
            
            # Save profile trace for analysis
            if args.output_dir:
                trace_path = os.path.join(args.output_dir, f"snn_profile_length_{length}.json")
                snn_prof.export_chrome_trace(trace_path)
                logger.info(f"Saved SNN profile trace to {trace_path}")

        except Exception as e:
            logger.error(f"Error profiling SNN model at sequence length {length}: {e}")
            pytest.fail(f"Error profiling SNN model: {e}")
            return False
    
    # Analyze results across all sequence lengths.
    #
    # IMPORTANT: this is a *software simulation*. TemporalSpikeProcessor evaluates the
    # network once per timestep, so the simulated SNN necessarily costs about T times the
    # ANN's wall-clock — it can never be "3x faster" here. Any energy advantage of a
    # spiking model is a property of event-driven neuromorphic hardware, which this repo
    # explicitly does not measure (see the README). Asserting a wall-clock speedup made
    # this test fail by construction on every run.
    #
    # What is meaningful to assert in simulation is that the temporal loop does not cost
    # more than its timestep count justifies. That is the regression this now guards.
    all_passed = True
    timesteps = max(1, int(getattr(args, 'timesteps', 1)))
    # Allowance over the ideal T-times-ANN cost, for wrapper and profiling overhead.
    overhead_allowance = float(getattr(args, 'simulation_overhead_allowance', 3.0))
    cost_budget = timesteps * overhead_allowance

    for length in test_lengths:
        ann_time = ann_metrics[length]['total_time_ms']
        snn_time = snn_metrics[length]['total_time_ms']

        cost_ratio = snn_time / max(ann_time, 0.001)
        efficiency_ratio = ann_time / max(snn_time, 0.001)

        # Report results
        logger.info(f"Sequence length {length}:")
        logger.info(f"  ANN time: {ann_time:.2f} ms")
        logger.info(f"  SNN time: {snn_time:.2f} ms")
        logger.info(f"  Simulated SNN cost: {cost_ratio:.2f}x ANN (T={timesteps}, budget {cost_budget:.1f}x)")
        logger.info(f"  Wall-clock ratio (informational, not an energy measurement): {efficiency_ratio:.2f}x")

        # Evaluate the cost budget per sequence length (this block must live inside the
        # loop; previously it sat outside and only judged the final length).
        if cost_ratio <= cost_budget:
            logger.info(f"  PASS: simulation cost {cost_ratio:.2f}x is within the {cost_budget:.1f}x budget for T={timesteps}")
        else:
            logger.error(
                f"  FAIL: simulation cost {cost_ratio:.2f}x exceeds the {cost_budget:.1f}x budget for T={timesteps} "
                "— the temporal wrapper is doing more work than its timestep count explains"
            )
            all_passed = False

        # Compare memory usage if available (reported for every length, not only on a timing FAIL).
        if device == 'cuda' and 'memory_mb' in ann_metrics[length] and 'memory_mb' in snn_metrics[length]:
            ann_memory = ann_metrics[length]['memory_mb']
            snn_memory = snn_metrics[length]['memory_mb']
            memory_reduction = (ann_memory - snn_memory) / ann_memory * 100 if ann_memory > 0 else 0

            logger.info(f"  Memory usage:")
            logger.info(f"    ANN: {ann_memory:.2f} MB")
            logger.info(f"    SNN: {snn_memory:.2f} MB")
            logger.info(f"    Reduction: {memory_reduction:.1f}%")

            # Memory efficiency target (SNN should use at least 20% less memory)
            memory_target = 20.0
            if memory_reduction >= memory_target:
                logger.info(f"    PASS: SNN uses {memory_reduction:.1f}% less memory (exceeds target of {memory_target:.1f}%)")
            else:
                logger.warning(f"    NOTICE: SNN uses only {memory_reduction:.1f}% less memory (below target of {memory_target:.1f}%)")

    # Save detailed metrics to file
    if args.output_dir:
        # Create the directory: only main() did, so any other caller (e.g. pytest) hit
        # FileNotFoundError here after the measurements had already been taken.
        os.makedirs(args.output_dir, exist_ok=True)
        metrics_path = os.path.join(args.output_dir, "energy_metrics.json")
        with open(metrics_path, 'w') as f:
            import json
            json.dump({
                'ann_metrics': ann_metrics,
                'snn_metrics': snn_metrics,
                'test_lengths': test_lengths,
                'device': device,
                'timesteps': timesteps,
                'simulation_cost_budget': cost_budget,
            }, f, indent=2)
        logger.info(f"Saved detailed energy metrics to {metrics_path}")
    
    if all_passed:
        logger.info(
            "PASS: test_energy_consumption (simulation cost within the T-timestep budget; "
            "this is a software-simulation cost check, not a hardware energy measurement)"
        )
    else:
        logger.error("FAIL: test_energy_consumption (simulation cost exceeds the T-timestep budget)")

    assert all_passed, "simulated SNN cost exceeded the T-timestep budget"
    return all_passed

def test_mixed_precision(model, tokenizer, args):
    """Validate the model can run in mixed precision mode for faster inference."""
    logger.info("Running: test_mixed_precision")
    device = args.device if hasattr(args, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Skip test if not on CUDA
    if device != 'cuda' or not torch.cuda.is_available():
        logger.info("Skipping mixed precision test as it requires CUDA")
        return True  # Not a failure, just skipped
    
    # Check if AMP is available.
    # NOTE: do not `import torch...` here. A function-local import binds the name `torch`
    # as a local for the WHOLE function, so the `torch.cuda.is_available()` call above
    # raised UnboundLocalError and this test could never run.
    if not hasattr(torch.cuda, "amp"):
        logger.warning("torch.cuda.amp not available, skipping mixed precision test")
        return True  # Not a failure, just skipped
    logger.info("torch.cuda.amp is available")

    # Create test input
    input_text = "Testing mixed precision inference"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    try:
        # Normal precision inference
        model.eval()
        with torch.no_grad():
            fp32_outputs = model(**inputs)
        
        fp32_dtype = fp32_outputs.logits.dtype if hasattr(fp32_outputs, 'logits') else fp32_outputs.dtype
        logger.info(f"Normal precision inference dtype: {fp32_dtype}")
        
        # Mixed precision inference
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                fp16_outputs = model(**inputs)
        
        fp16_dtype = fp16_outputs.logits.dtype if hasattr(fp16_outputs, 'logits') else fp16_outputs.dtype
        logger.info(f"Mixed precision inference dtype: {fp16_dtype}")
        
        # Verify that mixed precision actually used FP16
        is_mixed_precision = fp16_dtype in [torch.float16, torch.bfloat16]
        assert is_mixed_precision, f"Mixed precision inference didn't use FP16/BF16, got {fp16_dtype} instead"
        
        # Verify that outputs are reasonably close
        fp32_logits = fp32_outputs.logits if hasattr(fp32_outputs, 'logits') else fp32_outputs
        fp16_logits = fp16_outputs.logits if hasattr(fp16_outputs, 'logits') else fp16_outputs
        
        # Convert to same dtype for comparison
        fp32_logits = fp32_logits.to(torch.float32)
        fp16_logits = fp16_logits.to(torch.float32)
        
        # Calculate max absolute difference
        max_diff = torch.max(torch.abs(fp32_logits - fp16_logits)).item()
        logger.info(f"Max absolute difference between FP32 and mixed precision outputs: {max_diff}")
        
        # In machine learning, small precision differences are acceptable
        # The threshold depends on the specific model and application
        tolerance = 1e-2  # Reasonable tolerance for language models
        is_output_close = max_diff < tolerance
        
        if is_output_close:
            logger.info(f"PASS: Mixed precision outputs are within tolerance ({max_diff} < {tolerance})")
        else:
            logger.warning(f"NOTICE: Mixed precision outputs exceed tolerance ({max_diff} > {tolerance}), but may still be usable")
        
        # Calculate next token predictions with both precisions
        next_token_fp32 = torch.argmax(fp32_logits[0, -1, :]).item()
        next_token_fp16 = torch.argmax(fp16_logits[0, -1, :]).item()
        
        tokens_match = next_token_fp32 == next_token_fp16
        logger.info(f"Next token prediction: {'matches' if tokens_match else 'differs'} between precisions")
        if not tokens_match:
            logger.info(f"  FP32 predicted: {tokenizer.decode([next_token_fp32])}")
            logger.info(f"  FP16 predicted: {tokenizer.decode([next_token_fp16])}")
        
        # Time comparison
        logger.info("Comparing inference speed between precisions...")
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(**inputs)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    _ = model(**inputs)
        
        # Time FP32
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(10):
            with torch.no_grad():
                _ = model(**inputs)
        end_time.record()
        torch.cuda.synchronize()
        fp32_time_ms = start_time.elapsed_time(end_time) / 10
        
        # Time mixed precision
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(10):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    _ = model(**inputs)
        end_time.record()
        torch.cuda.synchronize()
        fp16_time_ms = start_time.elapsed_time(end_time) / 10
        
        speedup = fp32_time_ms / max(fp16_time_ms, 0.001)  # Avoid division by zero
        
        logger.info(f"FP32 inference time: {fp32_time_ms:.2f} ms")
        logger.info(f"Mixed precision inference time: {fp16_time_ms:.2f} ms")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Expect at least some speedup from mixed precision
        speedup_threshold = 1.2  # Expect at least 20% speedup
        is_faster = speedup >= speedup_threshold
        
        if is_faster:
            logger.info(f"PASS: Mixed precision provides sufficient speedup ({speedup:.2f}x > {speedup_threshold:.2f}x)")
        else:
            logger.warning(f"NOTICE: Mixed precision speedup is less than expected ({speedup:.2f}x < {speedup_threshold:.2f}x)")
        
        # Overall test result is based on:
        # 1. Mixed precision runs without errors
        # 2. It actually uses FP16 or BF16
        # 3. Outputs are within tolerance
        # We consider speedup as advisory but not a hard requirement
        
        test_passed = is_mixed_precision and (is_output_close or tokens_match)
        
        if test_passed:
            logger.info("PASS: test_mixed_precision")
        else:
            logger.error("FAIL: test_mixed_precision")
        
        return test_passed
    
    except Exception as e:
        logger.error(f"Error during mixed precision test: {e}")
        pytest.fail(f"Error during mixed precision test: {e}")
        return False

def test_loihi_compatibility(model, tokenizer, args):
    """Verify that the model is compatible with neuromorphic hardware like Intel Loihi."""
    logger.info("Running: test_loihi_compatibility")
    
    loihi_flag = getattr(model, "_is_loihi_compatible", False)
    if not loihi_flag or not hasattr(model, "_loihi_config"):
        msg = "Loihi export metadata missing; skipping hardware compatibility checks (simulation-only pipeline)."
        logger.warning(msg)
        pytest.skip(msg)
    
    # Check if Loihi-specific attributes are present
    loihi_config_present = hasattr(model, '_loihi_config')
    if loihi_config_present:
        logger.info("✓ Model has _loihi_config attribute")
        config = model._loihi_config
        
        # Validate required configuration parameters for Loihi
        required_params = ["neuron_model", "threshold", "core_mapping", "synapse_encoding", "weight_precision"]
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            logger.error(f"FAIL: Loihi config is missing required parameters: {missing_params}")
            loihi_config_present = False
        else:
            logger.info("✓ Loihi config has all required parameters")
            
            # Validate parameter values
            if config["neuron_model"] not in ["LIF", "IF", "AdaptiveLIF"]:
                logger.error(f"FAIL: Unsupported neuron model for Loihi: {config['neuron_model']}")
                loihi_config_present = False
            else:
                logger.info(f"✓ Neuron model {config['neuron_model']} is supported by Loihi")
                
            if config["synapse_encoding"] not in ["sparse", "dense"]:
                logger.error(f"FAIL: Unsupported synapse encoding: {config['synapse_encoding']}")
                loihi_config_present = False
            else:
                logger.info(f"✓ Synapse encoding {config['synapse_encoding']} is supported")
                
            if not isinstance(config["weight_precision"], int) or config["weight_precision"] not in [1, 2, 4, 8]:
                logger.error(f"FAIL: Unsupported weight precision: {config['weight_precision']}")
                loihi_config_present = False
            else:
                logger.info(f"✓ Weight precision {config['weight_precision']} bits is supported")
    else:
        logger.warning("NOTICE: Model does not have _loihi_config attribute")
    
    # Check if LIF neurons are used in the model
    lif_neurons_present = False
    lif_count = 0
    
    for name, module in model.named_modules():
        if "LIF" in module.__class__.__name__:
            lif_neurons_present = True
            lif_count += 1
            
            # Check if neuron parameters are Loihi-compatible
            if hasattr(module, "v_threshold"):
                # Loihi has limited threshold precision
                if isinstance(module.v_threshold, torch.Tensor) and module.v_threshold.numel() > 1:
                    logger.warning(f"NOTICE: Module {name} has per-channel thresholds which may not be directly mappable to Loihi")
                elif hasattr(module.v_threshold, "item") and module.v_threshold.item() <= 0:
                    logger.error(f"FAIL: Module {name} has non-positive threshold: {module.v_threshold.item()}")
                    lif_neurons_present = False
            else:
                logger.warning(f"NOTICE: Module {name} is missing v_threshold attribute")
                
            # Check for reset mechanisms
            if hasattr(module, "v_reset"):
                if module.v_reset is not None and module.v_reset != 0:
                    logger.warning(f"NOTICE: Module {name} has non-zero v_reset which may require adjustment for Loihi")
            
            # Check for time constants
            if hasattr(module, "tau"):
                # Loihi has limited time constant precision
                if isinstance(module.tau, torch.Tensor) and module.tau.numel() > 1:
                    logger.warning(f"NOTICE: Module {name} has per-channel time constants which may not be directly mappable to Loihi")
    
    if lif_count > 0:
        logger.info(f"✓ Found {lif_count} LIF neurons in the model")
    else:
        logger.error("FAIL: No LIF neurons found in the model")
        lif_neurons_present = False
        
    # Check for surrogate gradients which may not be needed on Loihi
    surrogate_gradients_present = False
    for name, module in model.named_modules():
        if "surrogate" in str(module.__class__).lower():
            surrogate_gradients_present = True
            logger.info(f"✓ Found surrogate gradient function in {name}: {module.__class__.__name__}")
            break
            
    if not surrogate_gradients_present:
        logger.warning("NOTICE: No surrogate gradient functions found in the model")
    
    # Check for sparse connectivity which is ideal for Loihi
    sparse_connectivity = False
    for name, param in model.named_parameters():
        if "weight" in name:
            sparsity = 1.0 - (torch.count_nonzero(param) / param.numel())
            if sparsity > 0.5:  # More than 50% zeros
                sparse_connectivity = True
                logger.info(f"✓ {name} has {sparsity:.1%} sparsity which is ideal for Loihi")
                break
                
    if not sparse_connectivity:
        logger.warning("NOTICE: Model lacks sparse connectivity which is recommended for Loihi")
    
    # Define minimum conditions for Loihi compatibility
    loihi_compatible = lif_neurons_present
    loihi_optimized = lif_neurons_present and (loihi_config_present or sparse_connectivity)
    
    # Final assessment
    if loihi_optimized:
        logger.info("PASS: test_loihi_compatibility (fully optimized)")
        return True
    elif loihi_compatible:
        logger.info("PASS: test_loihi_compatibility (compatible but not fully optimized)")
        # This is considered a pass since it can still run, just not optimally
        return True
    else:
        logger.error("FAIL: test_loihi_compatibility (not compatible)")
        assert False, "model is not Loihi compatible"
        return False


def test_loihi_constraints(model, args):
    """Simulation-time Loihi export-readiness checks (no hardware claims)."""
    logger.info("Running: test_loihi_constraints")
    export_ready, report = validate_loihi_export_readiness(model, intended_weight_bits=8)
    report_path = write_report(report, Path("local") / "loihi_constraints_reports")
    logger.info(f"Wrote Loihi constraints report: {report_path}")

    if export_ready:
        logger.info("PASS: test_loihi_constraints (no HARD_BLOCK findings)")
        return True

    hard_blocks = [f.get('id') for f in report.get('findings', []) if f.get('severity') == 'HARD_BLOCK']
    if not getattr(args, 'loihi_mode', False):
        # The default conversion keeps dense softmax attention (SpikeAttention), which is
        # not a Loihi-native primitive — so HARD_BLOCK findings are the *expected* result
        # here, not a regression. Demanding export-readiness without --loihi_mode made
        # `--test_all` fail by construction on every supported model.
        msg = (
            "Model was not converted in Loihi mode (--loihi_mode); export-readiness is "
            f"not expected. Findings: {hard_blocks}. Report: {report_path}"
        )
        logger.warning(f"SKIP: test_loihi_constraints — {msg}")
        pytest.skip(msg)
        return True

    logger.error("FAIL: test_loihi_constraints (HARD_BLOCK findings present; see report)")
    assert False, f"Loihi export constraints report contains HARD_BLOCK findings: {hard_blocks}"
    return False

def _run_cli_test(label, fn, *fn_args, **fn_kwargs):
    """
    Run one test function from the CLI and reduce it to a pass/fail boolean.

    The test helpers double as pytest tests and call pytest.skip()/pytest.fail(). Those
    raise OutcomeException, which derives from BaseException — so `except Exception` in
    main() did NOT catch them and a single skipped test (e.g. Loihi compatibility with no
    export metadata) aborted the whole `--test_all` run with a traceback, leaving the
    remaining tests unrun.
    """
    logger.info(f"Testing {label}")
    try:
        return bool(fn(*fn_args, **fn_kwargs))
    except Skipped as e:
        logger.warning(f"SKIP: {label}: {e}")
        return True  # a skip is not a failure
    except Failed as e:
        logger.error(f"FAIL: {label}: {e}")
        return False
    except AssertionError as e:
        logger.error(f"FAIL: {label}: {e}")
        return False


def main():
    logger.info("Entering main function...")
    args = parse_args()
    logger.info(f"Testing conversational capabilities with {args.model_name}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device  # Add device to args
    logger.info(f"Using device: {device}")
    
    try:
        # Step 1: Load the model and tokenizer
        logger.info(f"Loading model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name)

        # simplified_conversion() rewrites `base_model` IN PLACE (LayerNorm and attention
        # are swapped on the same object) and returns a wrapper around it. Passing
        # base_model as the "ANN reference" therefore compared the converted model with
        # itself, so the fidelity/parity tests measured nothing and always passed. Load a
        # separate, untouched copy when a parity test is going to run.
        ann_reference_model = None
        if args.test_all or args.test_fidelity or args.test_parity_multi_turn:
            logger.info("Loading a separate unconverted reference model for parity tests")
            ann_reference_model = AutoModelForCausalLM.from_pretrained(args.model_name)
            ann_reference_model.eval()

        if args.loihi_mode:
            # Marker used by simplified_conversion to swap attention implementation
            base_model._stac_loihi_mode = True
        if args.loihi_quantize:
            base_model._stac_loihi_quantize = True
        if args.loihi_prune:
            base_model._stac_loihi_prune = True
            base_model._stac_loihi_prune_target = float(args.loihi_prune_target)
        if args.loihi_embed_buckets and int(args.loihi_embed_buckets) > 0:
            base_model._stac_loihi_embed_buckets = int(args.loihi_embed_buckets)
        
        # Fix for tokenizer which doesn't have a pad token
        if tokenizer.pad_token is None:
            logger.info("Setting pad_token to eos_token for tokenizer")
            tokenizer.pad_token = tokenizer.eos_token
        
        # Step 2: Convert to SNN
        # Note: GELU->ReLU replacement degrades generation quality significantly
        # Skip it by default for coherence tests unless loihi_mode is enabled
        skip_gelu = getattr(args, 'skip_gelu_replacement', False) or not args.loihi_mode
        if skip_gelu:
            logger.info("Skipping GELU->ReLU replacement (preserves generation quality)")
        else:
            logger.info("Replacing GeLU activations with ReLU (required for neuromorphic deployment)")
            base_model = replace_gelu_with_relu(base_model)

        logger.info(f"Converting to SNN with T={args.timesteps}")
        base_model.T = args.timesteps
        snn_model = simplified_conversion(base_model, args.timesteps, skip_gelu_replacement=True)

        # Optional: load PEFT adapter into inner model for parity evaluation
        if args.adapter_dir:
            try:
                from peft import PeftModel
                if hasattr(snn_model, "snn_model"):
                    snn_model.snn_model = PeftModel.from_pretrained(snn_model.snn_model, args.adapter_dir)
                    logger.info(f"Loaded PEFT adapter into SNN inner model from: {args.adapter_dir}")
                    # Optional wrapper scalar
                    ls_path = Path(args.adapter_dir) / "logit_scale.pt"
                    if ls_path.exists() and hasattr(snn_model, "logit_scale"):
                        snn_model.logit_scale.data.copy_(torch.load(ls_path, map_location="cpu").to(snn_model.logit_scale.device))
                        logger.info(f"Loaded logit_scale from: {ls_path}")
                    # Optional SpikeLayerNorm sidecar
                    ln_path = Path(args.adapter_dir) / "spike_layernorm_state.pt"
                    if ln_path.exists():
                        try:
                            from smollm2_converter import SpikeLayerNorm
                            ln_state = torch.load(ln_path, map_location="cpu")
                            loaded = 0
                            for name, m in snn_model.named_modules():
                                if isinstance(m, SpikeLayerNorm) and name in ln_state:
                                    m.load_state_dict(ln_state[name], strict=True)
                                    loaded += 1
                            logger.info(f"Loaded SpikeLayerNorm state for {loaded} modules from: {ln_path}")
                        except Exception as e:
                            logger.warning(f"Found spike_layernorm_state.pt but failed to load it: {e}")
                else:
                    logger.warning("adapter_dir provided but SNN model has no snn_model attribute; skipping adapter load.")
            except Exception as e:
                logger.error(f"Failed to load adapter from {args.adapter_dir}: {e}")
                raise
        
        # Move to device
        snn_model = snn_model.to(device)
        
        # Step 4: Test conversational capabilities
        logger.info("Testing conversation with the SNN model")
        success = simulate_conversation(
            snn_model, 
            tokenizer, 
            turns=args.test_turns,
            device=device,
            max_context_length=args.max_context_length
        )
        
        # Run specific tests based on flags
        if args.test_all or args.test_position_boundaries:
            pos_success = _run_cli_test("position ID boundaries", test_position_id_boundaries, snn_model, tokenizer, args)
            success = success and pos_success
        
        if args.test_all or args.test_attention_mask:
            mask_success = _run_cli_test("attention mask continuity", test_attention_mask_continuity, snn_model, tokenizer, args)
            success = success and mask_success
        
        if args.test_all or args.test_multi_turn:
            multi_turn_success = _run_cli_test("multi-turn coherence", test_multi_turn_coherence, snn_model, tokenizer, args)
            success = success and multi_turn_success
        
        if args.test_all or args.test_energy:
            energy_success = _run_cli_test("energy consumption", test_energy_consumption, snn_model, tokenizer, args)
            success = success and energy_success

        if args.test_all or args.test_loihi_constraints:
            loihi_constraints_success = _run_cli_test("Loihi export constraints", test_loihi_constraints, snn_model, args)
            success = success and loihi_constraints_success

        if args.test_all or args.test_tsp_state:
            tsp_success = _run_cli_test("TemporalSpikeProcessor state retention", test_tsp_state_retention, snn_model, tokenizer, args)
            success = success and tsp_success

        if args.test_all or args.test_fidelity:
            fidelity_success = _run_cli_test("ANN<->SNN fidelity parity", test_fidelity_parity, ann_reference_model.to(device), snn_model, tokenizer, args)
            success = success and fidelity_success

        if args.test_all or args.test_parity_multi_turn:
            parity_success = _run_cli_test("ANN<->SNN multi-turn parity", test_multi_turn_parity, ann_reference_model.to(device), snn_model, tokenizer, args)
            success = success and parity_success
        
        # Test mixed precision (if supported)
        if args.test_all:
            mixed_precision_success = _run_cli_test("mixed precision", test_mixed_precision, snn_model, tokenizer, args)
            success = success and mixed_precision_success
        
        # Test Loihi compatibility (if supported)
        if args.test_all:
            loihi_success = _run_cli_test("Loihi compatibility", test_loihi_compatibility, snn_model, tokenizer, args)
            success = success and loihi_success
        
        # Step 5: Save the model if requested
        if success:
            logger.info(f"Saving SNN model to {args.output_dir}")
            save_snn_model(snn_model, tokenizer, args.output_dir)
            logger.info(f"SNN model saved to {args.output_dir}")

        if success:
            logger.info("All tests completed successfully!")
            return 0
        logger.error("One or more tests failed.")
        return 1

    except Exception as e:
        logger.error(f"Error during conversation test: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    logger.info("Executing from __main__...")
    sys.exit(main()) 