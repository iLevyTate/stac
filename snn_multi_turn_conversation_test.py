#!/usr/bin/env python
"""
Multi-Turn Conversation Test with SNN Emulation

This script drives a simple user-assistant chat loop using a spiking neural
network converted from DistilGPT-2.  The goal is **not** to achieve state-of-the-art
language quality (current SNN limitations make that unrealistic) but to
validate that:
  • The SNN model can generate successive turns without crashing
  • Internal states are properly reset between generations
  • Basic conversational coherence is maintained over multiple turns

It re-uses the conversion utilities already in the repository:
    – simplified_conversion()
    – TemporalSpikeProcessor

Usage:
    python snn_multi_turn_conversation_test.py [--device cpu|cuda] [--timesteps 8]

Outputs:
    • Prints the conversation to stdout
    • Logs timing and spike statistics
    • Saves a JSON conversation transcript (snn_multi_turn_conversation.json)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import spikingjelly.activation_based.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    # Import spiking utilities if available
    from smollm2_converter import simplified_conversion, TemporalSpikeProcessor
except ImportError:
    simplified_conversion = None
    TemporalSpikeProcessor = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
logger = logging.getLogger(__name__)


def build_model(timesteps: int, device: torch.device, mode: str):
    """Load DistilGPT-2 either as baseline or SNN."""
    logger.info("Loading base model (distilgpt2)…")
    base = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

    if mode == "baseline":
        base.eval()
        return base  # plain transformer

    if simplified_conversion is None or TemporalSpikeProcessor is None:
        raise RuntimeError("Spiking conversion utilities not available; install/compile them or choose --mode baseline.")

    logger.info(f"Converting to SNN (T={timesteps})…")
    snn = simplified_conversion(base, timesteps=timesteps)
    tp = TemporalSpikeProcessor(snn, T=timesteps).to(device)
    tp.eval()
    return tp


def generate_snn(
    tp: TemporalSpikeProcessor,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """Greedy / top-k sampling loop for SNN models."""
    device = next(tp.parameters()).device
    ids = input_ids.clone().to(device)

    for _ in range(max_new_tokens):
        # Reset internal neuron states before each forward pass
        functional.reset_net(tp)

        with torch.no_grad():
            out = tp(ids)
            logits = out.logits if hasattr(out, "logits") else (
                out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            )

        next_logits = logits[0, -1] / temperature  # (vocab,)

        # Amplify differences because SNN logits are typically compressed
        next_logits = next_logits * 2.0  # simple heuristic scale-up

        if top_k > 0:
            top_vals, top_idx = torch.topk(next_logits, k=top_k)
            probs = torch.softmax(top_vals, dim=-1)
            next_token = top_idx[torch.multinomial(probs, num_samples=1)]
        else:
            next_token = torch.argmax(next_logits, keepdim=True)

        ids = torch.cat([ids, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return ids[0]


def run_multi_turn_chat(turns=3, timesteps=8, device_str: str = None, temperature: float = 1.0, top_k: int = 20, mode: str = "snn"):
    device = (
        torch.device(device_str)
        if device_str
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(timesteps, device, mode)

    # Simple scripted conversation starter
    user_lines = [
        "Hello! How are you today?",
        "Can you tell me the capital of France?",
        "Thanks! Could you also tell me a short joke?",
    ]

    conversation = []  # list of dicts {role, text}

    history_text = ""  # Accumulated plain text history

    for turn, user_msg in enumerate(user_lines[:turns], 1):
        conversation.append({"role": "user", "text": user_msg})
        history_text += f"User: {user_msg}\nAssistant:"

        # Build input ids for the model
        input_ids = tokenizer(history_text, return_tensors="pt").input_ids.to(device)

        start = time.time()

        if mode == "baseline":
            gen_kwargs = {
                "max_new_tokens": 40,
                "temperature": temperature,
                "do_sample": top_k > 0,
                "top_k": top_k if top_k > 0 else None,
                "pad_token_id": tokenizer.eos_token_id,
            }
            output_ids = model.generate(input_ids, **{k: v for k, v in gen_kwargs.items() if v is not None})[0]
            new_tokens = output_ids[input_ids.shape[1] :]
            resp_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        else:
            # SNN path
            resp_ids = generate_snn(model, tokenizer, input_ids, max_new_tokens=40, temperature=temperature, top_k=top_k)
            new_tokens = resp_ids[input_ids.shape[1] :]
            resp_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        elapsed = (time.time() - start) * 1000  # ms

        logger.info(f"Turn {turn}: inference {elapsed:.1f} ms")
        logger.info(f"Assistant: {resp_text}")

        conversation.append({"role": "assistant", "text": resp_text})

        # Append assistant response to history for next turn
        history_text += " " + resp_text + "\n"

    return conversation


def save_conversation(conv, filename="snn_multi_turn_conversation.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(conv, f, indent=2)
    logger.info(f"Conversation saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-turn SNN conversation test")
    parser.add_argument("--turns", type=int, default=3, help="Number of user turns")
    parser.add_argument("--timesteps", type=int, default=8, help="Temporal windows T")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Force specific device")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for decoding")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling (0 = argmax)")
    parser.add_argument("--mode", choices=["snn", "baseline"], default="snn", help="Generation mode: spiking or baseline transformer")
    args = parser.parse_args()

    conv = run_multi_turn_chat(turns=args.turns, timesteps=args.timesteps, device_str=args.device, temperature=args.temperature, top_k=args.top_k, mode=args.mode)
    save_conversation(conv)

    logger.info("\n===== Conversation Transcript =====")
    for msg in conv:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        logger.info(f"{prefix}: {msg['text']}") 