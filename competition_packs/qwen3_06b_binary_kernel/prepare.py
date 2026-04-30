from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition_packs.qwen3_06b_shared.shared import (
    PRISM_BINARY_BONSAI_MODEL_ID,
    REFERENCE_17B_PARAMETER_COUNT,
    BenchmarkConfig,
    write_prepare_report,
)


CONFIG = BenchmarkConfig(
    pack_name="qwen3_06b_binary_kernel",
    quant_mode="binary",
    secondary_metric_name="speedup",
    quality_floor=0.76,
    allow_runtime_patch=True,
    reference_model_id=PRISM_BINARY_BONSAI_MODEL_ID,
    reference_parameter_count=REFERENCE_17B_PARAMETER_COUNT,
    native_bits_per_parameter=1.0,
)


def main() -> int:
    manifest = write_prepare_report(CONFIG, pack_dir=Path(__file__).resolve().parent)
    print(f"baseline_manifest={manifest.name}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
