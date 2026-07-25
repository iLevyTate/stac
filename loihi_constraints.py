"""
Loihi export-readiness constraints validator (simulation-time).

This module does NOT claim hardware execution. It provides evidence-driven checks that a
given SNN(-wrapped) model is structurally closer to Loihi-style deployment constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import torch


@dataclass(frozen=True)
class Finding:
    id: str
    severity: str  # "HARD_BLOCK" | "WARNING" | "INFO"
    message: str
    detail: Optional[Dict[str, Any]] = None


def _safe_class_name(obj: Any) -> str:
    try:
        return obj.__class__.__name__
    except Exception:
        return type(obj).__name__


def _count_params(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _param_dtype_hist(model: torch.nn.Module) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for p in model.parameters():
        k = str(p.dtype)
        hist[k] = hist.get(k, 0) + int(p.numel())
    return hist


def _weight_sparsity(model: torch.nn.Module, max_tensors: int = 50) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    count = 0
    for name, p in model.named_parameters():
        if "weight" not in name:
            continue
        if p.numel() == 0:
            continue
        if count >= max_tensors:
            break
        with torch.no_grad():
            nz = int(torch.count_nonzero(p).item())
            total = int(p.numel())
            sparsity = 1.0 - (nz / total)
        out.append({"name": name, "numel": total, "nonzero": nz, "sparsity": sparsity, "dtype": str(p.dtype)})
        count += 1
    return out


def _quantization_summary(model: torch.nn.Module) -> Dict[str, Any]:
    q_modules = 0
    q_buffers_int8 = 0
    total_modules = 0
    for _name, m in model.named_modules():
        total_modules += 1
        cn = _safe_class_name(m)
        if cn in ("QuantizedLinearLike", "QuantizedEmbedding", "QuantizedHashedEmbedding"):
            q_modules += 1
            for bname, buf in m.named_buffers(recurse=False):
                if isinstance(buf, torch.Tensor) and buf.dtype == torch.int8:
                    q_buffers_int8 += 1
    return {
        "quantized_module_count": q_modules,
        "quantized_int8_buffer_count": q_buffers_int8,
        "total_module_count": total_modules,
    }


def _quantized_weight_sparsity(model: torch.nn.Module, max_tensors: int = 50) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    count = 0
    for name, m in model.named_modules():
        cn = _safe_class_name(m)
        if cn not in ("QuantizedLinearLike", "QuantizedEmbedding", "QuantizedHashedEmbedding"):
            continue
        if not hasattr(m, "qweight"):
            continue
        q = getattr(m, "qweight")
        if not isinstance(q, torch.Tensor) or q.numel() == 0:
            continue
        if count >= max_tensors:
            break
        with torch.no_grad():
            nz = int(torch.count_nonzero(q).item())
            total = int(q.numel())
            sparsity = 1.0 - (nz / total)
        out.append({"name": f"{name}.qweight", "numel": total, "nonzero": nz, "sparsity": sparsity, "dtype": str(q.dtype)})
        count += 1
    return out


def _is_stateful_spiking_neuron(module: Any) -> bool:
    """
    True for modules that actually integrate membrane state and emit spikes.

    Checked structurally rather than by class name: SpikingJelly neurons subclass
    `neuron.BaseNode` and expose a voltage threshold plus membrane state; the STAC V1
    AdEx neuron carries V_th/V_reset/V_rest parameters.
    """
    try:
        from spikingjelly.activation_based import neuron as _sj_neuron
        if isinstance(module, _sj_neuron.BaseNode):
            return True
    except Exception:
        pass
    if _safe_class_name(module) == "DLPFCAdExNeuron":
        return True
    # Generic fallback: a threshold plus some form of membrane state.
    has_threshold = hasattr(module, "v_threshold") or hasattr(module, "V_th")
    has_state = hasattr(module, "v") or hasattr(module, "v_reset") or hasattr(module, "V_reset")
    return bool(has_threshold and has_state)


def _count_invoked_spiking_neurons(
    model: torch.nn.Module,
    neuron_modules: List[torch.nn.Module],
    sample_input: torch.Tensor,
) -> Optional[int]:
    """
    Run one forward pass and count how many of `neuron_modules` were actually invoked.

    Presence in the module tree is not participation: a model can construct spiking
    neurons and then bypass them in forward(), which no static check can see.

    Hooks are attached to the module objects directly. Resolving them by *name* does not
    work here: the names are collected from the unwrapped inner model
    ("transformer.h.0.attn.q_spk") while the forward pass runs on the wrapper, whose
    namespace prefixes them ("snn_model.transformer.h.0.attn.q_spk"). Name matching
    therefore hooked nothing and this always returned 0, regardless of the model.

    Returns None if the forward pass could not be run.
    """
    invoked = set()
    handles = []

    def _hook(mod, _inp, _out):
        invoked.add(id(mod))

    try:
        for module in neuron_modules:
            handles.append(module.register_forward_hook(_hook))
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(sample_input)
        if was_training:
            model.train()
    except Exception:
        return None
    finally:
        for h in handles:
            h.remove()
    return len(invoked)


def _is_embedding_weight(name: str) -> bool:
    # Common HF GPT-2 embeddings
    return (
        name.endswith("transformer.wte.weight")
        or name.endswith("transformer.wpe.weight")
        or name.endswith("transformer.wte.qweight")
        or name.endswith("transformer.wpe.qweight")
        or ".wte.weight" in name
        or ".wpe.weight" in name
        or ".wte.qweight" in name
        or ".wpe.qweight" in name
    )


def validate_loihi_export_readiness(
    model: torch.nn.Module,
    *,
    intended_weight_bits: int = 8,
    require_spiking_neurons: bool = True,
    sample_input: Optional[torch.Tensor] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (export_ready, report_dict).

    export_ready is True only when there are no HARD_BLOCK findings.

    Pass `sample_input` (a token-id tensor) to additionally check that the spiking
    neurons are actually *invoked* during a forward pass, rather than merely present in
    the module tree.
    """
    findings: List[Finding] = []

    # Unwrap common wrapper patterns
    inner = getattr(model, "snn_model", None)
    wrapper_name = _safe_class_name(model)
    if inner is not None and isinstance(inner, torch.nn.Module):
        findings.append(Finding(
            id="wrapper_detected",
            severity="INFO",
            message=(
                f"Model appears to be wrapped ({wrapper_name}); structural checks below "
                "run against the inner snn_model."
            ),
            detail={"wrapper": wrapper_name, "inner": _safe_class_name(inner)},
        ))
    else:
        inner = model

    # Spiking neuron presence.
    #
    # Counting by class name alone treated SpikeSoftmax (a plain torch.softmax) and
    # SpikeAttention (whose LIF neurons are constructed but bypassed) as evidence of
    # spiking, so a model that performs no spiking computation at all was reported as
    # "Detected N spiking-like modules". Separate the two questions: which modules are
    # *named* like spiking components, and which are actual stateful spiking neurons.
    name_matched = []
    real_neurons = []
    real_neuron_modules = []
    for _n, m in inner.named_modules():
        cn = _safe_class_name(m)
        if "LIF" in cn or "IFNode" in cn or "Spike" in cn or "AdEx" in cn:
            name_matched.append(_n)
        if _is_stateful_spiking_neuron(m):
            real_neurons.append(_n)
            real_neuron_modules.append(m)

    if require_spiking_neurons and not real_neurons:
        findings.append(Finding(
            id="no_spiking_modules_detected",
            severity="HARD_BLOCK",
            message=(
                "No stateful spiking neurons found. "
                f"{len(name_matched)} module(s) are named like spiking components but do not "
                "carry membrane state, so the model performs no spiking computation."
            ),
            detail={"name_matched": name_matched[:20], "name_matched_count": len(name_matched)},
        ))
    else:
        findings.append(Finding(
            id="spiking_modules_detected",
            severity="INFO",
            message=(
                f"Detected {len(real_neurons)} stateful spiking neuron module(s). "
                "Presence in the module tree does not prove they run; pass sample_input "
                "to check participation in the forward pass."
            ),
            detail={
                "spiking_neurons": real_neurons[:20],
                "spiking_neuron_count": len(real_neurons),
                "name_matched_count": len(name_matched),
            },
        ))

    # Are those neurons actually on the forward path?
    if real_neurons and sample_input is not None:
        invoked = _count_invoked_spiking_neurons(model, real_neuron_modules, sample_input)
        if invoked is None:
            findings.append(Finding(
                id="spiking_participation_unknown",
                severity="INFO",
                message="Could not run a forward pass to check spiking-neuron participation.",
            ))
        elif invoked == 0:
            findings.append(Finding(
                id="spiking_neurons_not_invoked",
                severity="HARD_BLOCK",
                message=(
                    f"None of the {len(real_neurons)} spiking neurons were invoked during a "
                    "forward pass: they are constructed but bypassed, so the model performs "
                    "no spiking computation."
                ),
                detail={"spiking_neuron_count": len(real_neurons), "invoked": 0},
            ))
        else:
            findings.append(Finding(
                id="spiking_neurons_invoked",
                severity="INFO" if invoked == len(real_neurons) else "WARNING",
                message=f"{invoked}/{len(real_neurons)} spiking neurons ran during a forward pass.",
                detail={"spiking_neuron_count": len(real_neurons), "invoked": invoked},
            ))

    # Attention is the big blocker for direct Loihi mapping in this repo today.
    #
    # - SpikeAttention (as used in V2) is still dense attention under the hood.
    # - HuggingFace backbones (e.g., GPT2Attention) are also dense attention.
    spike_attention_modules = []
    dense_attention_modules = []
    for name, m in inner.named_modules():
        cn = _safe_class_name(m)
        if cn == "SpikeAttention":
            spike_attention_modules.append(name)
        elif "Attention" in cn or cn.endswith("Attn") or cn.endswith("MHA"):
            dense_attention_modules.append({"name": name, "class": cn})

    if spike_attention_modules:
        findings.append(Finding(
            id="dense_attention_present_spikeattention",
            severity="HARD_BLOCK",
            message="SpikeAttention modules present. Dense QK^T attention/softmax is not a Loihi-native primitive; requires a dedicated mapping strategy.",
            detail={"modules": spike_attention_modules[:20], "count": len(spike_attention_modules)},
        ))
    if dense_attention_modules:
        findings.append(Finding(
            id="dense_attention_present_backbone",
            severity="HARD_BLOCK",
            message="Dense attention modules detected (e.g., HF Attention). Direct Loihi mapping typically requires a dedicated attention mapping strategy.",
            detail={"examples": dense_attention_modules[:20], "count": len(dense_attention_modules)},
        ))

    # Quantization / dtype readiness (heuristic)
    dtype_hist = _param_dtype_hist(inner)
    float_params = sum(v for k, v in dtype_hist.items() if "float" in k)
    qsum = _quantization_summary(inner)
    if qsum["quantized_module_count"] > 0:
        findings.append(Finding(
            id="fake_int8_quantization_detected",
            severity="INFO",
            message="Detected fake int8 quantization wrappers (simulation-time evidence).",
            detail=qsum,
        ))
    if float_params > 0:
        findings.append(Finding(
            id="float_parameters_present",
            severity="WARNING" if intended_weight_bits in (8, 4, 2, 1) else "INFO",
            message="Model parameters are floating-point in simulation; Loihi deployment typically requires integer/fixed-point quantization.",
            detail={"dtype_histogram_numel": dtype_hist, "intended_weight_bits": intended_weight_bits},
        ))

    # Loihi metadata presence (only indicates someone tried to export/map)
    # Check the wrapper AND the inner model: an exporter may tag either one, and looking
    # only at the wrapper reported "metadata missing" for a correctly tagged inner model.
    has_loihi_config = hasattr(model, "_loihi_config") or hasattr(inner, "_loihi_config")
    has_loihi_flag = bool(
        getattr(model, "_is_loihi_compatible", False)
        or getattr(inner, "_is_loihi_compatible", False)
    )
    if has_loihi_flag and has_loihi_config:
        findings.append(Finding(
            id="loihi_metadata_present",
            severity="INFO",
            message="Model has Loihi export metadata flags (_is_loihi_compatible + _loihi_config).",
        ))
    else:
        findings.append(Finding(
            id="loihi_metadata_missing",
            severity="INFO",
            message="Loihi export metadata not present. This is expected for simulation-only runs.",
        ))

    # Sparsity reporting (not a pass/fail gate)
    sparsity_list = _weight_sparsity(inner)
    quant_sparsity_list = _quantized_weight_sparsity(inner)
    if sparsity_list:
        dense_hotspots = [w for w in sparsity_list if w["sparsity"] < 0.1 and not _is_embedding_weight(w["name"])]
        dense_embeddings = [w for w in sparsity_list if w["sparsity"] < 0.1 and _is_embedding_weight(w["name"])]
        if dense_embeddings:
            findings.append(Finding(
                id="dense_embeddings_present",
                severity="INFO",
                message="Dense embedding tables detected (expected in many LMs; may need special handling on neuromorphic targets).",
                detail={"examples": dense_embeddings[:5]},
            ))
        if dense_hotspots:
            findings.append(Finding(
                id="dense_weight_hotspots",
                severity="WARNING",
                message="Dense weight tensors detected (low sparsity). Loihi benefits from sparse connectivity.",
                detail={"examples": dense_hotspots[:10]},
            ))
        else:
            findings.append(Finding(
                id="sparsity_ok",
                severity="INFO",
                message="No extremely dense hotspots detected in sampled weight tensors.",
            ))

    if quant_sparsity_list:
        dense_q_hotspots = [w for w in quant_sparsity_list if w["sparsity"] < 0.1 and not _is_embedding_weight(w["name"])]
        dense_q_embeddings = [w for w in quant_sparsity_list if w["sparsity"] < 0.1 and _is_embedding_weight(w["name"])]
        if dense_q_embeddings:
            findings.append(Finding(
                id="dense_quantized_embeddings_present",
                severity="INFO",
                message="Dense quantized embedding tables detected (expected in many LMs; may need special handling on neuromorphic targets).",
                detail={"examples": dense_q_embeddings[:5]},
            ))
        if dense_q_hotspots:
            findings.append(Finding(
                id="dense_quantized_weight_hotspots",
                severity="WARNING",
                message="Dense quantized weight tensors detected (low sparsity). Consider pruning/sparsification for Loihi efficiency.",
                detail={"examples": dense_q_hotspots[:10]},
            ))
        else:
            findings.append(Finding(
                id="quantized_sparsity_ok",
                severity="INFO",
                message="No extremely dense hotspots detected in sampled quantized weight tensors.",
            ))

    hard_blocks = [f for f in findings if f.severity == "HARD_BLOCK"]
    export_ready = len(hard_blocks) == 0

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "wrapper": wrapper_name,
        "inner_model": _safe_class_name(inner),
        "param_count": _count_params(inner),
        "intended_weight_bits": intended_weight_bits,
        "export_ready": export_ready,
        "hard_block_count": len(hard_blocks),
        "warning_count": sum(1 for f in findings if f.severity == "WARNING"),
        "findings": [asdict(f) for f in findings],
        "sampled_weight_sparsity": sparsity_list,
        "sampled_quantized_weight_sparsity": quant_sparsity_list,
    }
    return export_ready, report


def write_report(report: Dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = report.get("timestamp_utc", datetime.now(timezone.utc).isoformat()).replace(":", "").replace("-", "")
    path = out_dir / f"loihi_constraints_{ts}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


