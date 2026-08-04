# Changelog

All notable changes to STAC are recorded here. Versioning is milestone-based and tracked with
git tags; the corresponding Zenodo deposit shares the same version.

## [4.0.0] — 2026-08-04

A correctness-and-honesty release. An audit established that STAC's spiking pathway had never
actually functioned as published; this version fixes the underlying bugs, makes conversion
faithful, and replaces overstated claims with measured results. The headline outcome: **STAC's
conversion is now trustworthy, and the record now matches what the code does.**

### Fixed — the spiking pathway now functions

- **STAC V1 dead spiking pathway.** The AdEx neurons emitted zero spikes for every input
  (membrane pinned ~14.7 mV below threshold; required drive 75 vs. observed 2.4), and the
  Gaussian surrogate gradient underflowed float32 to exactly zero, so no gradient reached the
  spiking layer and the L1 sparsity term was identically zero. Fixed by a `CurrentDrive` layer
  and sizing the surrogate to the membrane scale. Reproduction: `scripts/verify_v1_corrigendum.py`.
- **STAC V2 rotary position embeddings dropped.** `SpikeAttention` never applied RoPE; converting
  a Llama-family model (every SmolLM2 variant) discarded all positional information — 19–28× worse
  perplexity with spiking off. Now applied. Conversion-only perplexity on SmolLM2-135M/360M went
  from 19–28× worse to **1.00×** (`tests/test_rope_fidelity.py`).
- **STAC V2 attention spike encoding.** Replaced the fixed-threshold hard-reset LIF (a 1-bit
  quantiser: corr 0.805 with magnitude) with a calibrated signed soft-reset IF (0.989) — 5.7× less
  damaging.
- **RMSNorm handling.** The normalization pass matched only `nn.LayerNorm`, silently skipping
  Llama-family RMSNorm; added `SpikeRMSNorm`.
- Numerous earlier audit fixes (state-dict corruption, NaN on padded batches, scale-blind
  quantization, cache handling, silent CLI failures) — see PR #11.

### Added — measurement, tooling, and tests

- `spike_metrics.py` — spike-count telemetry and an operation-level energy projection (45nm
  Horowitz figures), the analysis the README previously referenced but did not contain.
- `scripts/energy_analysis.py` — the coverage/energy crossover analysis.
- `scripts/coverage_quality_sweep.py`, `scripts/logit_calibration_probe.py` — the conversion-
  quality studies.
- `scripts/finetune_spiking.py` — spike-aware fine-tuning harness (the recovery path).
- `spike_coverage.py` — calibrated signed soft-reset IF encoding for extending spike coverage.
- `scripts/make_test_models.py` — offline tiny GPT-2/Llama/GQA fixtures, so the suite runs with
  no network.
- New tests: `test_energy_analysis.py`, `test_spike_coverage.py`, `test_liveness.py`,
  `test_rope_fidelity.py`, `test_v1_baseline.py`, `test_device_safety.py`.

### Findings documented (see `docs/`)

- **Energy is coverage-bound, not sparsity-bound** (`energy-crossover.md`): `E_SNN/E_ANN =
  T·(ρ·f·r + 1 − f)`, validated to 0.1%. The current design (~5% coverage) projects **7.6× worse**
  than the dense ANN.
- **Post-hoc conversion of a frozen model collapses it** (`coverage-quality.md`): near-constant
  output; every frozen-weight remedy (coverage, calibration, leak, timesteps) fails; damage
  localized to the first two blocks and precision-independent.
- **Training recovers quality** (proof of concept): 10.5× perplexity recovery in 300 CPU steps.
- **A methodological warning**: standard metrics (spike sparsity, cosine-to-teacher) report their
  best value on a dead network; evaluation needs spike counts and prediction diversity.
- `findings-summary.md` ties the whole study together.

### Changed — claims brought in line with measurements

- README, `stac_v1/README.md`, `docs/snn_conversion_fixes.md`, `docs/conversion_workflow.md`, and
  `docs/api_reference.md` corrected: removed "successful spiking training", "retaining multi-turn
  conversational ability", "energy efficiency", and "mathematically equivalent" framings where
  they overstated the measured position.
- Added `CITATION.cff` and `.zenodo.json` (structured citation and deposition metadata; neither
  existed before).

### Correction to the published record

- `docs/corrigendum-2026-07.md` and `docs/paper-corrigendum-submission.md` document a correction
  to the accompanying book chapter: the V1 spiking-mechanism claims describe intended rather than
  observed behaviour, and the software citation pointed at a version (`2.0.0.3`) that contained no
  V1 implementation — now resolved by this `4.0.0` release.

### Not included

- Physical neuromorphic-hardware measurements (energy figures remain operation-count projections).
- A conclusive spike-aware fine-tuning run at scale (T=8, longer context) — requires GPU hardware.

## [3.0.0-beta] — 2025-12-22

Repository reorganization; consolidated the V1 notebook into the `stac_v1/` package. (The spiking
pathway was still inert at this tag — see 4.0.0.)
