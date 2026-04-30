from __future__ import annotations

from typing import Any

from competition_packs.qwen3_06b_shared.shared import default_submission


def build_submission(*, seed: int, time_budget_seconds: int, debug_dataset_name: str | None = None) -> dict[str, Any]:
    return default_submission(quant_mode="binary", kernel_task=True)

