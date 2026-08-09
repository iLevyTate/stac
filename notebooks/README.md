# Notebooks

## `stac_v2_colab.ipynb` — try STAC V2 in Google Colab

A runnable, zero-setup tour of the V2 conversion pipeline. Clone-free: open it in Colab and
run top to bottom.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iLevyTate/stac/blob/claude/codebase-updates-v2-status-p19lcw/notebooks/stac_v2_colab.ipynb)

What it walks through:

1. Install the pinned dependencies (`transformers < 4.48`, `numpy < 2.0`).
2. Run the test suite, including the liveness checks that guard the fixed spiking pathway.
3. Convert DistilGPT-2 with spiking **off** — verify it reproduces the original model.
4. Convert with spiking **on** — see real spikes, the (unfavourable) energy projection, and
   the timestep-dependent drift that a frozen model cannot recover from.
5. The coverage-vs-sparsity energy analysis and its scaling behaviour.
6. (Optional) SmolLM2-135M, showing the RoPE/RMSNorm fix.
7. **End-to-end retraining** (GPU recommended): convert → extend coverage → calibrate →
   backprop-through-time training with distillation → before/after perplexity → save the
   model, then reload it and generate text. This is the full "does training fix it?" loop
   the audit couldn't finish for lack of GPU hardware. Note it installs the `datasets`
   package (WikiText-2), which the base requirements leave optional.

> **Expectation setting:** with spiking off, conversion is faithful; with spiking on and the
> weights frozen, the model collapses and the projected energy is worse than the dense ANN.
> Training (step 7) is the demonstrated path to a usable spiking model. See
> [`docs/findings-summary.md`](../docs/findings-summary.md).

If Colab downgrades `numpy`/`transformers` and asks you to restart, do **Runtime → Restart
session**, then re-run the install cell and continue.
