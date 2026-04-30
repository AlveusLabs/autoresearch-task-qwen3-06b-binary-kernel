from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

MODEL_ID = "Qwen/Qwen3-0.6B"
TOKENIZER_ID = MODEL_ID
QWEN3_17B_MODEL_ID = "Qwen/Qwen3-1.7B"
PRISM_BINARY_BONSAI_MODEL_ID = "prism-ml/Bonsai-1.7B-gguf"
PRISM_TERNARY_BONSAI_MODEL_ID = "prism-ml/Ternary-Bonsai-1.7B-gguf"
REFERENCE_17B_PARAMETER_COUNT = 1_720_000_000
PACK_SEED = 20260422
PUBLIC_DEBUG_DATASETS = ("pack-local-proxy",)
MAX_HIGH_PRECISION_FRACTION = 0.10
FLOAT_BITS = 16
BASELINE_REFERENCE_BITS = 16
REFERENCE_RUNTIME_REPEATS = 24
PRIMARY_METRIC_NAME = "heldout_ppl"
MAX_PARAMETER_COUNT_MULTIPLIER = 16
PPL_RESOLUTION_NATS = 0.02

CORPUS_TEXTS: tuple[str, ...] = (
    "Qwen style proxy benchmarks should reward real tradeoffs instead of score reporting tricks.",
    "Binary compression needs hard eligibility gates before any Pareto frontier matters.",
    "Ternary compression can keep quality high when zero thresholds and scale choices are tuned carefully.",
    "Kernel tasks are only meaningful when the validator owns both correctness and timing measurement.",
    "A hidden rotating shard is more useful than a giant public corpus when the validator must stay cheap.",
    "The public debug datasets help miners iterate locally but do not decide reward distribution.",
    "Compression ratio should count the actual inference time representation, not just a prose claim.",
    "Reproducible artifacts are mandatory because validators must rerun the same benchmark twice.",
    "Shape validity matters because the task keeps the same model interface and tensor contract.",
    "Only a tiny high precision allowance should survive when the competition is mostly binary or ternary.",
    "Validator owned scoring prevents miners from smuggling in fake heldout metrics.",
    "Runtime speedups only count if the output stays aligned with the reference quantized inference path.",
    "Backend policy should choose the shard handle while public code stays generic.",
    "A stable pack is better than a huge brittle benchmark that nobody can replay cheaply.",
    "Pareto ranking means quality and efficiency both matter after hard filters are passed.",
    "Centerless tasks reward the idea proposer and the implementer once a better result appears.",
    "Standard tasks keep the reward path simpler because only validator replay decides the score.",
    "Quality floors stop obviously broken artifacts from gaming the compression side of the frontier.",
    "A small validator owned proxy is enough to exercise the control flow end to end.",
    "Artifacts should encode exactly which tensors stay high precision and which are quantized.",
    "Binary kernels often win by packing signs densely and reducing Python overhead.",
    "Ternary kernels have to deal with zeros efficiently or the speedup disappears.",
    "Hidden shard rotation should be cheap to change from the backend without editing pack code.",
    "A good compression plan usually preserves a few sensitive tensors and quantizes the rest aggressively.",
    "Heldout quality should be computed from the benchmarked artifact instead of the miner summary.",
    "The fixed tokenizer and I O contract mean submissions cannot redefine the task itself.",
    "Validator logs should make the gating decisions obvious when a submission is rejected.",
    "Cheap validation is only useful if the benchmark still respects the intended economics.",
    "Inference time representations should be measured directly so hidden side tensors are not free.",
    "Kernel benchmarks need fixed traces, fixed iteration counts, and fixed reference implementations.",
    "The benchmark can be small as long as the rules are explicit and the measurements are owned by validators.",
    "Public task repos should be live only when the benchmark pack already exists and runs cleanly.",
)


@dataclass(frozen=True, slots=True)
class LayerSpec:
    name: str
    rows: int
    cols: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def element_count(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    pack_name: str
    quant_mode: str
    secondary_metric_name: str
    quality_floor: float
    allow_runtime_patch: bool = False
    reference_model_id: str | None = None
    reference_parameter_count: int | None = None
    native_bits_per_parameter: float | None = None
    max_non_native_fraction: float = MAX_HIGH_PRECISION_FRACTION
    ppl_resolution_nats: float = PPL_RESOLUTION_NATS


@dataclass(frozen=True, slots=True)
class ReferenceLimits:
    model_id: str
    parameter_count: int
    baseline_bits: int
    baseline_size_bytes: int
    max_parameter_count: int
    max_compressed_bits: int
    native_bits_per_parameter: float
    max_non_native_fraction: float


VOCAB = tuple(sorted({token for text in CORPUS_TEXTS for token in text.lower().split()}))
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
INPUT_DIM = len(VOCAB)
LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec("embed_proj", INPUT_DIM, 48),
    LayerSpec("attn_out", 48, 48),
    LayerSpec("mlp_up", 48, 96),
    LayerSpec("mlp_down", 96, 48),
    LayerSpec("lm_head", 48, INPUT_DIM),
)


def _seed_from_text(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _sorted_layer_specs() -> tuple[LayerSpec, ...]:
    return LAYER_SPECS


def _make_base_weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(PACK_SEED)
    weights: dict[str, np.ndarray] = {}
    for spec in _sorted_layer_specs():
        scale = 0.11 if spec.name == "lm_head" else 0.08
        weights[spec.name] = rng.normal(0.0, scale, size=spec.shape).astype(np.float64)
    return weights


BASE_WEIGHTS = _make_base_weights()
TOTAL_WEIGHT_ELEMENTS = sum(spec.element_count for spec in _sorted_layer_specs())
TOTAL_BASELINE_BITS = TOTAL_WEIGHT_ELEMENTS * BASELINE_REFERENCE_BITS
TOTAL_BASELINE_BYTES = math.ceil(TOTAL_BASELINE_BITS / 8)
MAX_COMPRESSED_BITS = TOTAL_BASELINE_BITS
MAX_PARAMETER_COUNT = TOTAL_WEIGHT_ELEMENTS * MAX_PARAMETER_COUNT_MULTIPLIER


def _default_native_bits(quant_mode: str) -> float:
    if quant_mode == "binary":
        return 1.0
    if quant_mode == "ternary":
        return 2.0
    return float(BASELINE_REFERENCE_BITS)


def _reference_limits(config: BenchmarkConfig) -> ReferenceLimits:
    if config.reference_parameter_count is None:
        return ReferenceLimits(
            model_id=MODEL_ID,
            parameter_count=int(TOTAL_WEIGHT_ELEMENTS),
            baseline_bits=int(TOTAL_BASELINE_BITS),
            baseline_size_bytes=int(TOTAL_BASELINE_BYTES),
            max_parameter_count=int(MAX_PARAMETER_COUNT),
            max_compressed_bits=int(MAX_COMPRESSED_BITS),
            native_bits_per_parameter=_default_native_bits(config.quant_mode),
            max_non_native_fraction=float(MAX_HIGH_PRECISION_FRACTION),
        )

    reference_parameter_count = int(config.reference_parameter_count)
    if reference_parameter_count <= 0:
        raise ValueError("reference_parameter_count must be positive")
    native_bits_per_parameter = (
        _default_native_bits(config.quant_mode)
        if config.native_bits_per_parameter is None
        else float(config.native_bits_per_parameter)
    )
    if not math.isfinite(native_bits_per_parameter) or native_bits_per_parameter <= 0.0:
        raise ValueError("native_bits_per_parameter must be positive and finite")
    max_non_native_fraction = float(config.max_non_native_fraction)
    if not math.isfinite(max_non_native_fraction) or not 0.0 <= max_non_native_fraction <= 1.0:
        raise ValueError("max_non_native_fraction must be in [0, 1]")

    baseline_bits = reference_parameter_count * BASELINE_REFERENCE_BITS
    max_compressed_bits = int(
        math.ceil(
            reference_parameter_count
            * (
                ((1.0 - max_non_native_fraction) * native_bits_per_parameter)
                + (max_non_native_fraction * FLOAT_BITS)
            )
        )
    )
    return ReferenceLimits(
        model_id=config.reference_model_id or MODEL_ID,
        parameter_count=reference_parameter_count,
        baseline_bits=int(baseline_bits),
        baseline_size_bytes=int(math.ceil(baseline_bits / 8)),
        max_parameter_count=int(reference_parameter_count * MAX_PARAMETER_COUNT_MULTIPLIER),
        max_compressed_bits=max_compressed_bits,
        native_bits_per_parameter=float(native_bits_per_parameter),
        max_non_native_fraction=max_non_native_fraction,
    )


def _document_matrix(documents: list[str]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for text in documents:
        counts = np.zeros((INPUT_DIM,), dtype=np.float64)
        tokens = [token for token in text.lower().split() if token in TOKEN_TO_ID]
        for token in tokens:
            counts[TOKEN_TO_ID[token]] += 1.0
        if tokens:
            counts /= max(1, len(tokens))
        rows.append(counts)
    if not rows:
        raise RuntimeError("no documents selected for benchmark shard")
    return np.stack(rows, axis=0)


def _choose_documents(*, dataset_name: str, split_name: str, rotation_key: str, count: int = 16) -> list[str]:
    seed = _seed_from_text(f"{dataset_name}:{split_name}:{rotation_key}:{PACK_SEED}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(CORPUS_TEXTS))
    return [CORPUS_TEXTS[int(index)] for index in order[:count]]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _forward(weights: dict[str, np.ndarray], inputs: np.ndarray) -> np.ndarray:
    hidden = np.tanh(inputs @ weights["embed_proj"])
    hidden = np.tanh(hidden @ weights["attn_out"])
    hidden = np.tanh(hidden @ weights["mlp_up"])
    hidden = np.tanh(hidden @ weights["mlp_down"])
    return hidden @ weights["lm_head"]


def _teacher_probs(inputs: np.ndarray) -> np.ndarray:
    return _softmax(_forward(BASE_WEIGHTS, inputs))


def _quality_metrics(candidate_probs: np.ndarray, target_probs: np.ndarray) -> dict[str, float]:
    safe_target = np.clip(target_probs, 1e-9, 1.0)
    safe_candidate = np.clip(candidate_probs, 1e-9, 1.0)
    mean_cross_entropy = float(np.mean(np.sum(-safe_target * np.log(safe_candidate), axis=1)))
    mean_entropy = float(np.mean(np.sum(-safe_target * np.log(safe_target), axis=1)))
    mean_kl = float(mean_cross_entropy - mean_entropy)
    return {
        "heldout_ppl": float(math.exp(min(mean_cross_entropy, 700.0))),
        "heldout_cross_entropy_nats": mean_cross_entropy,
        "heldout_teacher_entropy_nats": mean_entropy,
        "heldout_kl": mean_kl,
        "heldout_relative_ppl": float(math.exp(min(mean_kl, 700.0))),
        "heldout_quality": float(math.exp(-mean_kl)),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _artifact_hash(artifact: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(artifact).encode("utf-8")).hexdigest()


def _artifact_manifest_size_bytes(artifact: dict[str, Any]) -> int:
    return len(_canonical_json(artifact).encode("utf-8"))


def _non_negative_int(value: Any, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _positive_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return parsed


def _validate_declared_accounting(artifact: dict[str, Any], summary: dict[str, Any]) -> None:
    accounting = artifact.get("accounting")
    if accounting is None:
        return
    if not isinstance(accounting, dict):
        raise ValueError("artifact.accounting must be an object")

    int_fields = {
        "parameter_count": "parameter_count",
        "compressed_bits": "compressed_bits",
        "compressed_size_bytes": "compressed_size_bytes",
        "high_precision_count": "total_high_precision_count",
        "rescue_count": "total_high_precision_count",
    }
    for declared_key, computed_key in int_fields.items():
        if declared_key not in accounting:
            continue
        declared = _non_negative_int(accounting[declared_key], field_name=f"artifact.accounting.{declared_key}")
        computed = int(summary[computed_key])
        if declared != computed:
            raise ValueError(
                f"artifact.accounting.{declared_key}={declared} does not match validator computed {computed}"
            )

    float_fields = {
        "overall_high_precision_fraction": "overall_high_precision_fraction",
        "high_precision_fraction": "overall_high_precision_fraction",
        "rescue_fraction": "overall_high_precision_fraction",
    }
    for declared_key, computed_key in float_fields.items():
        if declared_key not in accounting:
            continue
        try:
            declared = float(accounting[declared_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"artifact.accounting.{declared_key} must be a finite number") from exc
        computed = float(summary[computed_key])
        if not math.isfinite(declared) or not math.isclose(declared, computed, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                f"artifact.accounting.{declared_key}={declared:.12g} does not match "
                f"validator computed {computed:.12g}"
            )


def _accounting_entries_from_artifact(artifact: dict[str, Any], plans: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    accounting = artifact.get("accounting")
    if accounting is not None:
        if not isinstance(accounting, dict):
            raise ValueError("artifact.accounting must be an object")
        raw_entries = accounting.get("extra_entries", [])
        if not isinstance(raw_entries, list):
            raise ValueError("artifact.accounting.extra_entries must be a list")
        for entry in raw_entries:
            if not isinstance(entry, dict):
                raise ValueError("artifact.accounting.extra_entries entries must be objects")
            entries.append(dict(entry, _source="artifact.accounting.extra_entries"))

    for layer_name, plan in plans.items():
        raw_entries = plan.get("extra_components", plan.get("accounting_entries", []))
        if raw_entries is None:
            continue
        if not isinstance(raw_entries, list):
            raise ValueError(f"artifact layer {layer_name} extra_components must be a list")
        for entry in raw_entries:
            if not isinstance(entry, dict):
                raise ValueError(f"artifact layer {layer_name} extra_components entries must be objects")
            entries.append(dict(entry, layer=layer_name, _source=f"artifact.layers[{layer_name}].extra_components"))
    return entries


def _summarize_extra_accounting_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    allowed_roles = {
        "binary",
        "ternary",
        "quantized",
        "residual",
        "scale",
        "side",
        "metadata",
        "high_precision",
        "full_precision",
        "rescue",
    }
    rows: list[dict[str, Any]] = []
    total_parameter_count = 0
    total_compressed_bits = 0
    total_high_precision_count = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("artifact accounting entries must be objects")
        source = str(entry.get("_source") or "artifact accounting")
        name = str(entry.get("name") or f"extra_{index}").strip() or f"extra_{index}"
        role = str(entry.get("role") or "side").strip().lower()
        if role not in allowed_roles:
            raise ValueError(f"{source}.{name} uses unsupported accounting role: {role}")
        parameter_count = _non_negative_int(
            entry.get("parameter_count"),
            field_name=f"{source}.{name}.parameter_count",
        )
        if parameter_count == 0:
            raise ValueError(f"{source}.{name}.parameter_count must be greater than zero")

        if "compressed_bits" in entry:
            compressed_bits = _non_negative_int(
                entry["compressed_bits"],
                field_name=f"{source}.{name}.compressed_bits",
            )
        elif "size_bytes" in entry:
            compressed_bits = 8 * _non_negative_int(
                entry["size_bytes"],
                field_name=f"{source}.{name}.size_bytes",
            )
        elif "bits_per_parameter" in entry:
            bits_per_parameter = _positive_float(
                entry["bits_per_parameter"],
                field_name=f"{source}.{name}.bits_per_parameter",
            )
            compressed_bits = int(math.ceil(parameter_count * bits_per_parameter))
        else:
            raise ValueError(
                f"{source}.{name} must declare compressed_bits, size_bytes, or bits_per_parameter"
            )

        counts_toward_rescue = bool(entry.get("counts_toward_rescue", False)) or role in {
            "high_precision",
            "full_precision",
            "rescue",
        }
        high_precision_count = parameter_count if counts_toward_rescue else 0
        total_parameter_count += parameter_count
        total_compressed_bits += compressed_bits
        total_high_precision_count += high_precision_count
        rows.append(
            {
                "name": name,
                "role": role,
                "parameter_count": int(parameter_count),
                "compressed_bits": int(compressed_bits),
                "compressed_size_bytes": int(math.ceil(compressed_bits / 8)),
                "high_precision_count": int(high_precision_count),
                "counts_toward_rescue": bool(counts_toward_rescue),
                **({"layer": str(entry["layer"])} if "layer" in entry else {}),
            }
        )
    return {
        "entries": rows,
        "parameter_count": int(total_parameter_count),
        "compressed_bits": int(total_compressed_bits),
        "high_precision_count": int(total_high_precision_count),
    }


def _layer_plan_lookup(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = artifact.get("layers")
    if not isinstance(rows, list):
        raise ValueError("artifact.layers must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("artifact.layers rows must be objects")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("artifact layer entry is missing a name")
        by_name[name] = dict(raw)
    return by_name


def _quantize_layer(
    matrix: np.ndarray,
    *,
    mode: str,
    high_precision_fraction: float,
    scale_multiplier: float,
    threshold_multiplier: float,
    binary_basis_count: int = 1,
    binary_basis_scales: list[float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    flat = matrix.reshape(-1)
    keep_fraction = max(0.0, min(1.0, float(high_precision_fraction)))
    keep_count = int(round(len(flat) * keep_fraction))
    basis_count = max(1, int(binary_basis_count))

    if mode == "binary" and basis_count > 1:
        provided_scales: list[float] | None = None
        if binary_basis_scales is not None:
            provided_scales = [float(value) for value in binary_basis_scales]
            if len(provided_scales) != basis_count:
                raise ValueError(
                    f"binary_basis_scales length {len(provided_scales)} does not match "
                    f"binary_basis_count {basis_count}"
                )
            if not all(math.isfinite(value) and value > 0.0 for value in provided_scales):
                raise ValueError("binary_basis_scales values must be finite positive numbers")
        candidate = np.zeros_like(flat)
        residual = flat.copy()
        scales: list[float] = []
        for basis_index in range(basis_count):
            if provided_scales is None:
                scale = (float(np.mean(np.abs(residual))) or 1.0) * max(0.25, float(scale_multiplier))
            else:
                scale = provided_scales[basis_index]
            basis = np.where(residual >= 0.0, scale, -scale)
            candidate += basis
            residual -= basis
            scales.append(float(scale))
        keep_indices = np.array([], dtype=np.int64)
        if keep_count > 0:
            keep_indices = np.argpartition(np.abs(residual), -keep_count)[-keep_count:]
            candidate[keep_indices] = flat[keep_indices]
        quantized = candidate.reshape(matrix.shape)
        compressed_bits = (len(flat) * basis_count) + (keep_count * FLOAT_BITS) + (64 * basis_count)
        quantized_parameter_count = len(flat) * basis_count
        side_parameter_count = basis_count
        parameter_count = quantized_parameter_count + keep_count + side_parameter_count
        summary = {
            "high_precision_fraction": float(keep_count / max(1, len(flat))),
            "high_precision_count": int(keep_count),
            "quantized_parameter_count": int(quantized_parameter_count),
            "side_parameter_count": int(side_parameter_count),
            "parameter_count": int(parameter_count),
            "compressed_bits": int(compressed_bits),
            "compressed_size_bytes": int(math.ceil(compressed_bits / 8)),
            "scale": float(scales[0] if scales else 1.0),
            "scales": scales,
            "binary_basis_count": int(basis_count),
            "threshold": 0.0,
            "zero_count": 0,
        }
        return quantized, summary

    keep_indices = np.array([], dtype=np.int64)
    if keep_count > 0:
        keep_indices = np.argpartition(np.abs(flat), -keep_count)[-keep_count:]
    keep_mask = np.zeros((len(flat),), dtype=bool)
    if keep_indices.size:
        keep_mask[keep_indices] = True

    candidate = np.zeros_like(flat)
    if np.any(~keep_mask):
        quantized_source = flat[~keep_mask]
        abs_mean = float(np.mean(np.abs(quantized_source))) or 1.0
        scale = abs_mean * max(0.25, float(scale_multiplier))
        if mode == "binary":
            candidate[~keep_mask] = np.where(quantized_source >= 0.0, scale, -scale)
            zero_count = 0
            quant_bits = 1
        elif mode == "ternary":
            threshold = abs_mean * max(0.10, float(threshold_multiplier))
            signs = np.where(quantized_source >= 0.0, scale, -scale)
            candidate[~keep_mask] = np.where(np.abs(quantized_source) >= threshold, signs, 0.0)
            zero_count = int(np.sum(candidate[~keep_mask] == 0.0))
            quant_bits = 2
        else:
            raise ValueError(f"unsupported quant mode: {mode}")
    else:
        scale = 1.0
        threshold = 0.0
        zero_count = 0
        quant_bits = 1 if mode == "binary" else 2

    candidate[keep_mask] = flat[keep_mask]
    quantized = candidate.reshape(matrix.shape)
    compressed_bits = ((len(flat) - keep_count) * quant_bits) + (keep_count * FLOAT_BITS) + 64
    quantized_parameter_count = len(flat) - keep_count
    side_parameter_count = 1
    parameter_count = quantized_parameter_count + keep_count + side_parameter_count
    summary = {
        "high_precision_fraction": float(keep_count / max(1, len(flat))),
        "high_precision_count": int(keep_count),
        "quantized_parameter_count": int(quantized_parameter_count),
        "side_parameter_count": int(side_parameter_count),
        "parameter_count": int(parameter_count),
        "compressed_bits": int(compressed_bits),
        "compressed_size_bytes": int(math.ceil(compressed_bits / 8)),
        "scale": float(scale),
        "binary_basis_count": int(basis_count if mode == "binary" else 1),
        "threshold": float(threshold if mode == "ternary" else 0.0),
        "zero_count": int(zero_count),
    }
    return quantized, summary


def build_candidate_weights(config: BenchmarkConfig, artifact: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference_limits = _reference_limits(config)
    plans = _layer_plan_lookup(artifact)
    candidate: dict[str, np.ndarray] = {}
    layer_summaries: list[dict[str, Any]] = []
    total_high_precision_count = 0
    total_compressed_bits = 0
    total_parameter_count = 0
    for spec in _sorted_layer_specs():
        if spec.name not in plans:
            raise ValueError(f"artifact is missing layer: {spec.name}")
        plan = plans[spec.name]
        declared_shape = tuple(int(value) for value in plan.get("shape", spec.shape))
        if declared_shape != spec.shape:
            raise ValueError(f"shape mismatch for {spec.name}: {declared_shape} != {spec.shape}")
        quantized, summary = _quantize_layer(
            BASE_WEIGHTS[spec.name],
            mode=config.quant_mode,
            high_precision_fraction=float(plan.get("high_precision_fraction", 0.0)),
            scale_multiplier=float(plan.get("scale_multiplier", 1.0)),
            threshold_multiplier=float(plan.get("threshold_multiplier", 0.80)),
            binary_basis_count=int(plan.get("binary_basis_count", 1)),
            binary_basis_scales=plan.get("binary_basis_scales"),
        )
        candidate[spec.name] = quantized
        total_high_precision_count += int(summary["high_precision_count"])
        total_compressed_bits += int(summary["compressed_bits"])
        total_parameter_count += int(summary["parameter_count"])
        layer_summaries.append(
            {
                "name": spec.name,
                "shape": list(spec.shape),
                **summary,
            }
        )
    extra_accounting = _summarize_extra_accounting_entries(_accounting_entries_from_artifact(artifact, plans))
    total_high_precision_count += int(extra_accounting["high_precision_count"])
    total_compressed_bits += int(extra_accounting["compressed_bits"])
    total_parameter_count += int(extra_accounting["parameter_count"])
    overall_high_precision_fraction = total_high_precision_count / max(1, reference_limits.parameter_count)
    proxy_high_precision_fraction = total_high_precision_count / max(1, TOTAL_WEIGHT_ELEMENTS)
    compressed_size_bytes = int(math.ceil(total_compressed_bits / 8))
    summary = {
        "layers": layer_summaries,
        "extra_accounting_entries": extra_accounting["entries"],
        "total_high_precision_count": int(total_high_precision_count),
        "overall_high_precision_fraction": float(overall_high_precision_fraction),
        "proxy_high_precision_fraction": float(proxy_high_precision_fraction),
        "parameter_count": int(total_parameter_count),
        "parameter_count_multiplier": float(total_parameter_count / max(1, reference_limits.parameter_count)),
        "proxy_parameter_count_multiplier": float(total_parameter_count / max(1, TOTAL_WEIGHT_ELEMENTS)),
        "compressed_bits": int(total_compressed_bits),
        "compressed_size_bytes": int(compressed_size_bytes),
        "compression_ratio": float(reference_limits.baseline_bits / max(1, total_compressed_bits)),
        "baseline_parameter_count": int(reference_limits.parameter_count),
        "baseline_bits": int(reference_limits.baseline_bits),
        "baseline_size_bytes": int(reference_limits.baseline_size_bytes),
        "max_parameter_count": int(reference_limits.max_parameter_count),
        "max_compressed_bits": int(reference_limits.max_compressed_bits),
        "reference_model_id": reference_limits.model_id,
        "native_bits_per_parameter": float(reference_limits.native_bits_per_parameter),
        "max_non_native_fraction": float(reference_limits.max_non_native_fraction),
        "proxy_model_id": MODEL_ID,
        "proxy_parameter_count": int(TOTAL_WEIGHT_ELEMENTS),
    }
    _validate_declared_accounting(artifact, summary)
    return candidate, summary


def default_submission(*, quant_mode: str, kernel_task: bool) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for spec in _sorted_layer_specs():
        if spec.name in {"embed_proj", "lm_head"}:
            high_precision_fraction = 0.035 if quant_mode == "binary" else 0.025
        elif spec.name == "attn_out":
            high_precision_fraction = 0.010
        else:
            high_precision_fraction = 0.0
        layers.append(
            {
                "name": spec.name,
                "shape": list(spec.shape),
                "high_precision_fraction": high_precision_fraction,
                "scale_multiplier": 0.95 if kernel_task else 1.0,
                "threshold_multiplier": 0.70 if quant_mode == "ternary" else 1.0,
            }
        )
    return {
        "artifact_version": 1,
        "model_id": MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "quant_mode": quant_mode,
        "layers": layers,
    }


def _load_submission(train_module_name: str, *, seed: int, time_budget_seconds: int) -> dict[str, Any]:
    artifact_path = str(os.environ.get("AUTORESEARCH_SUBMISSION_ARTIFACT_PATH") or "").strip()
    if artifact_path:
        path = Path(artifact_path)
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"submission artifact not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"submission artifact must be JSON: {path}") from exc
        if not isinstance(artifact, dict):
            raise ValueError("submission artifact JSON must be an object")
        return artifact

    module = importlib.import_module(train_module_name)
    module = importlib.reload(module)
    builder = getattr(module, "build_submission", None)
    if builder is None:
        raise ValueError(f"{train_module_name} must define build_submission(...)")
    artifact = builder(
        seed=seed,
        time_budget_seconds=time_budget_seconds,
        debug_dataset_name=os.environ.get("AUTORESEARCH_DEBUG_DATASET", ""),
    )
    if not isinstance(artifact, dict):
        raise ValueError("build_submission(...) must return a dict artifact")
    return artifact


def _load_runtime(runtime_module_name: str | None):
    if not runtime_module_name:
        return None
    module = importlib.import_module(runtime_module_name)
    return importlib.reload(module)


def _reference_apply_layer(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rows = [row @ weights for row in inputs]
    return np.stack(rows, axis=0)


def _evaluate_runtime(candidate_weights: dict[str, np.ndarray], *, runtime_module_name: str | None) -> dict[str, Any]:
    if runtime_module_name is None:
        return {"speedup": None, "reference_seconds": None, "candidate_seconds": None}
    module = _load_runtime(runtime_module_name)
    apply_quantized_layer = getattr(module, "apply_quantized_layer", None)
    if apply_quantized_layer is None:
        raise ValueError(f"{runtime_module_name} must define apply_quantized_layer(inputs, weights)")

    rng = np.random.default_rng(PACK_SEED + 701)
    trace = rng.normal(0.0, 1.0, size=(32, 48)).astype(np.float64)
    candidate_seconds = 0.0
    reference_seconds = 0.0
    for spec in _sorted_layer_specs():
        if spec.rows != trace.shape[1]:
            continue
        weights = candidate_weights[spec.name]
        started = time.perf_counter()
        reference_output = None
        for _ in range(REFERENCE_RUNTIME_REPEATS):
            reference_output = _reference_apply_layer(trace, weights)
        reference_seconds += time.perf_counter() - started

        started = time.perf_counter()
        candidate_output = None
        for _ in range(REFERENCE_RUNTIME_REPEATS):
            candidate_output = np.asarray(apply_quantized_layer(trace, weights), dtype=np.float64)
        candidate_seconds += time.perf_counter() - started

        if reference_output is None or candidate_output is None:
            raise ValueError("runtime benchmark produced no output")
        if candidate_output.shape != reference_output.shape:
            raise ValueError(f"runtime output shape mismatch for {spec.name}")
        if not np.allclose(candidate_output, reference_output, atol=1e-6, rtol=1e-6):
            raise ValueError(f"runtime output drifted for {spec.name}")
        trace = np.tanh(candidate_output)

    speedup = reference_seconds / max(candidate_seconds, 1e-9)
    return {
        "speedup": float(speedup),
        "reference_seconds": float(reference_seconds),
        "candidate_seconds": float(candidate_seconds),
    }


def write_prepare_report(config: BenchmarkConfig, *, pack_dir: Path) -> Path:
    reference_limits = _reference_limits(config)
    report = {
        "pack_name": config.pack_name,
        "model_id": reference_limits.model_id,
        "reference_model_id": reference_limits.model_id,
        "proxy_model_id": MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "public_debug_datasets": list(PUBLIC_DEBUG_DATASETS),
        "metric_name": PRIMARY_METRIC_NAME,
        "metric_direction": "minimize",
        "ppl_resolution_nats": config.ppl_resolution_nats,
        "max_high_precision_fraction": reference_limits.max_non_native_fraction,
        "max_non_native_fraction": reference_limits.max_non_native_fraction,
        "native_bits_per_parameter": reference_limits.native_bits_per_parameter,
        "baseline_parameter_count": reference_limits.parameter_count,
        "baseline_size_bytes": reference_limits.baseline_size_bytes,
        "max_parameter_count": reference_limits.max_parameter_count,
        "max_compressed_bits": reference_limits.max_compressed_bits,
        "quality_floor": config.quality_floor,
        "secondary_metric_name": config.secondary_metric_name,
        "layer_specs": [
            {"name": spec.name, "shape": list(spec.shape), "element_count": spec.element_count}
            for spec in _sorted_layer_specs()
        ],
    }
    path = pack_dir / "baseline_manifest.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_benchmark(
    config: BenchmarkConfig,
    *,
    train_module_name: str,
    runtime_module_name: str | None = None,
    result_path: Path,
) -> int:
    dataset_name = str(os.environ.get("AUTORESEARCH_HELDOUT_DATASET") or "pack-local-proxy")
    dataset_mix = str(os.environ.get("AUTORESEARCH_HELDOUT_DATASET_MIX") or "").strip()
    split_name = str(os.environ.get("AUTORESEARCH_HELDOUT_SPLIT") or "validator-heldout")
    rotation_key = str(os.environ.get("AUTORESEARCH_HELDOUT_ROTATION_KEY") or "default-rotation")
    dataset_selection_key = dataset_mix or dataset_name
    documents = _choose_documents(dataset_name=dataset_selection_key, split_name=split_name, rotation_key=rotation_key)
    inputs = _document_matrix(documents)
    target_probs = _teacher_probs(inputs)

    first_artifact = _load_submission(train_module_name, seed=PACK_SEED, time_budget_seconds=90)
    second_artifact = _load_submission(train_module_name, seed=PACK_SEED, time_budget_seconds=90)
    first_hash = _artifact_hash(first_artifact)
    second_hash = _artifact_hash(second_artifact)
    if first_hash != second_hash:
        raise SystemExit("validator_rejection=reproducibility artifact hash changed between identical runs")

    candidate_weights, candidate_summary = build_candidate_weights(config, first_artifact)
    high_precision_fraction = float(candidate_summary["overall_high_precision_fraction"])
    max_high_precision_fraction = float(candidate_summary["max_non_native_fraction"])
    if high_precision_fraction > max_high_precision_fraction + 1e-12:
        raise SystemExit(
            f"validator_rejection=high_precision_fraction high_precision_fraction={high_precision_fraction:.6f}"
        )
    parameter_count = int(candidate_summary["parameter_count"])
    max_parameter_count = int(candidate_summary["max_parameter_count"])
    if parameter_count > max_parameter_count:
        raise SystemExit(
            "validator_rejection=parameter_count "
            f"parameter_count={parameter_count} max_parameter_count={max_parameter_count}"
        )
    compressed_bits = int(candidate_summary["compressed_bits"])
    max_compressed_bits = int(candidate_summary["max_compressed_bits"])
    if compressed_bits > max_compressed_bits:
        raise SystemExit(
            "validator_rejection=compressed_size "
            f"compressed_bits={compressed_bits} max_compressed_bits={max_compressed_bits}"
        )

    quality_metrics = _quality_metrics(_softmax(_forward(candidate_weights, inputs)), target_probs)
    quality = float(quality_metrics["heldout_quality"])
    heldout_ppl = float(quality_metrics[PRIMARY_METRIC_NAME])
    if quality < config.quality_floor:
        raise SystemExit(f"validator_rejection=quality_floor heldout_quality={quality:.6f}")

    runtime_summary = _evaluate_runtime(candidate_weights, runtime_module_name=runtime_module_name)
    secondary_value = (
        float(candidate_summary["compression_ratio"])
        if config.secondary_metric_name == "compression_ratio"
        else float(runtime_summary["speedup"])
    )
    eligibility = {
        "shape_valid": True,
        "high_precision_valid": True,
        "parameter_count_valid": True,
        "compressed_size_valid": True,
        "accounting_valid": True,
        "reproducible": True,
        "quality_floor_passed": True,
    }
    downloaded_artifact_bytes_raw = str(os.environ.get("AUTORESEARCH_SUBMISSION_ARTIFACT_BYTES") or "").strip()
    downloaded_artifact_bytes = int(downloaded_artifact_bytes_raw) if downloaded_artifact_bytes_raw else None
    artifact_manifest_bytes = _artifact_manifest_size_bytes(first_artifact)
    metrics = {
        PRIMARY_METRIC_NAME: heldout_ppl,
        "heldout_quality": quality,
        "heldout_cross_entropy_nats": float(quality_metrics["heldout_cross_entropy_nats"]),
        "heldout_teacher_entropy_nats": float(quality_metrics["heldout_teacher_entropy_nats"]),
        "heldout_kl": float(quality_metrics["heldout_kl"]),
        "heldout_relative_ppl": float(quality_metrics["heldout_relative_ppl"]),
        "compression_ratio": float(candidate_summary["compression_ratio"]),
        "parameter_count": float(candidate_summary["parameter_count"]),
        "parameter_count_multiplier": float(candidate_summary["parameter_count_multiplier"]),
        "proxy_parameter_count_multiplier": float(candidate_summary["proxy_parameter_count_multiplier"]),
        "compressed_bits": float(candidate_summary["compressed_bits"]),
        "compressed_size_bytes": float(candidate_summary["compressed_size_bytes"]),
        "baseline_parameter_count": float(candidate_summary["baseline_parameter_count"]),
        "baseline_size_bytes": float(candidate_summary["baseline_size_bytes"]),
        "high_precision_count": float(candidate_summary["total_high_precision_count"]),
        "high_precision_fraction": high_precision_fraction,
        "proxy_high_precision_fraction": float(candidate_summary["proxy_high_precision_fraction"]),
        "artifact_manifest_bytes": float(artifact_manifest_bytes),
        "submission_artifact_bytes": float(downloaded_artifact_bytes) if downloaded_artifact_bytes is not None else None,
        "speedup": float(runtime_summary["speedup"]) if runtime_summary["speedup"] is not None else None,
    }
    report = {
        "pack_name": config.pack_name,
        "model_id": str(candidate_summary["reference_model_id"]),
        "reference_model_id": str(candidate_summary["reference_model_id"]),
        "proxy_model_id": MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "ppl_resolution_nats": float(config.ppl_resolution_nats),
        "native_bits_per_parameter": float(candidate_summary["native_bits_per_parameter"]),
        "max_non_native_fraction": float(candidate_summary["max_non_native_fraction"]),
        "metric_name": PRIMARY_METRIC_NAME,
        "metric_value": heldout_ppl,
        "metrics": metrics,
        "eligibility": eligibility,
        "artifact_hash": first_hash,
        "secondary_metric_name": config.secondary_metric_name,
        "secondary_metric_value": secondary_value,
        "high_precision_fraction": high_precision_fraction,
        "parameter_count": int(candidate_summary["parameter_count"]),
        "parameter_count_multiplier": float(candidate_summary["parameter_count_multiplier"]),
        "proxy_parameter_count_multiplier": float(candidate_summary["proxy_parameter_count_multiplier"]),
        "compressed_bits": int(candidate_summary["compressed_bits"]),
        "compressed_size_bytes": int(candidate_summary["compressed_size_bytes"]),
        "baseline_parameter_count": int(candidate_summary["baseline_parameter_count"]),
        "baseline_size_bytes": int(candidate_summary["baseline_size_bytes"]),
        "max_parameter_count": int(candidate_summary["max_parameter_count"]),
        "max_compressed_bits": int(candidate_summary["max_compressed_bits"]),
        "artifact_manifest_bytes": int(artifact_manifest_bytes),
        "submission_artifact_bytes": downloaded_artifact_bytes,
        "layer_summaries": candidate_summary["layers"],
        "extra_accounting_entries": candidate_summary["extra_accounting_entries"],
        "dataset_handle": {
            "dataset": dataset_name,
            "dataset_mix": dataset_mix or None,
            "split": split_name,
            "rotation_key_hash": hashlib.sha256(rotation_key.encode("utf-8")).hexdigest()[:16],
        },
        "runtime": runtime_summary,
    }
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"heldout_ppl={heldout_ppl:.6f}")  # noqa: T201
    print(f"heldout_quality={quality:.6f}")  # noqa: T201
    print(f"compression_ratio={candidate_summary['compression_ratio']:.6f}")  # noqa: T201
    print(f"parameter_count={candidate_summary['parameter_count']}")  # noqa: T201
    print(f"compressed_size_bytes={candidate_summary['compressed_size_bytes']}")  # noqa: T201
    if runtime_summary["speedup"] is not None:
        print(f"speedup={runtime_summary['speedup']:.6f}")  # noqa: T201
    print(f"high_precision_fraction={high_precision_fraction:.6f}")  # noqa: T201
    return 0
