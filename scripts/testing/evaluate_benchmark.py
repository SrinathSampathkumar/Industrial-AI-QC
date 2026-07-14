"""Re-evaluate an existing calibrated Week-2 raw-score report.

This is a lightweight validation entry point for CI or report regeneration. It
uses the same calibration-aware evaluation function as ``benchmark_patchcore``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.registry.model_registry import ModelRegistry
from scripts.testing.benchmark_patchcore import (
    DEFAULT_RESULTS_OUTPUT,
    DEFAULT_WEEK2_RAW_OUTPUT,
    evaluate_calibrated_scores,
    load_calibration,
    read_raw_predictions,
    verify_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and regenerate calibrated Week-2 metrics.")
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_WEEK2_RAW_OUTPUT)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_OUTPUT)
    args = parser.parse_args()

    calibration = load_calibration()
    registry = ModelRegistry()
    verify_registry(registry, calibration, load_models=False)
    raw = read_raw_predictions(args.raw_input)
    _, results = evaluate_calibrated_scores(raw, registry, calibration)
    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_output, index=False)
    print(f"Validated {len(results)} calibrated categories and wrote {args.results_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
