"""Fast contract tests for category-routed PatchCore inspection.

The production checkpoints are deliberately replaced with a small
PatchCore-compatible test model.  Loading all 15 heavyweight checkpoints in a
unit-test job is not CI-safe; this suite instead exercises the real category
router and inference implementation, including OpenCV decoding, score
normalization, heatmap writing, and the returned response contract.

Run with:
    pytest tests/test_patchcore_categories.py -q
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.api import category_router
from scripts.inference import inference_patchcore
from scripts.registry.model_registry import ModelRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)
RESPONSE_FIELDS = {
    "category",
    "prediction",
    "raw_score",
    "normalized_score",
    "threshold",
    "heatmap_path",
}
MAX_SUITE_RUNTIME_SECONDS = 60


class _PatchcoreTestDouble:
    """Minimal PatchCore output contract used only to keep CI model-free."""

    def __init__(self, category: str) -> None:
        self.category = category

    def __call__(self, image_tensor: torch.Tensor) -> SimpleNamespace:
        _, _, height, width = image_tensor.shape
        # A category-specific but deterministic score catches wrong-category
        # routing without requiring a 15-checkpoint model load.
        score = 0.10 + (CATEGORIES.index(self.category) / 100)
        return SimpleNamespace(
            pred_score=torch.tensor([score], dtype=torch.float32),
            pred_label=torch.tensor([False]),
            anomaly_map=torch.zeros((1, 1, height, width), dtype=torch.float32),
        )


class _ScoreScalerTestDouble:
    """Avoid joblib/scikit-learn artifact loading in the fast contract suite."""

    def normalize(self, score: float, category: str) -> float:
        del category
        return float(score * 100)


@pytest.fixture(autouse=True)
def fast_patchcore_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list[str]]:
    """Route requests into the real inference function using test-only artifacts."""
    loaded_categories: list[str] = []

    def fake_is_trained(_: ModelRegistry, category: str) -> bool:
        return category in CATEGORIES

    def fake_load(_: ModelRegistry, category: str) -> _PatchcoreTestDouble:
        loaded_categories.append(category)
        return _PatchcoreTestDouble(category)

    def isolated_predict(category: str, image_path: str) -> dict:
        return inference_patchcore.predict_image(
            category=category,
            image_path=image_path,
            output_dir=str(tmp_path / "heatmaps"),
        )

    monkeypatch.setattr(ModelRegistry, "is_trained", fake_is_trained)
    monkeypatch.setattr(ModelRegistry, "load", fake_load)
    monkeypatch.setattr(inference_patchcore, "ScoreScaler", _ScoreScalerTestDouble)
    monkeypatch.setattr(category_router, "predict_image", isolated_predict)
    return {"loaded_categories": loaded_categories}


@pytest.fixture(scope="session", autouse=True)
def execution_summary(request: pytest.FixtureRequest) -> None:
    """Print a compact CI summary and fail the suite if it exceeds 60 seconds."""
    started = time.perf_counter()
    yield
    elapsed = time.perf_counter() - started
    reporter = request.config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        passed = len(reporter.stats.get("passed", []))
        failed = len(reporter.stats.get("failed", [])) + len(reporter.stats.get("error", []))
        reporter.write_line(
            f"Industrial-AI-QC test summary: passed={passed} failed={failed} "
            f"execution_time={elapsed:.2f}s"
        )
    assert elapsed < MAX_SUITE_RUNTIME_SECONDS, (
        f"Test suite exceeded {MAX_SUITE_RUNTIME_SECONDS} seconds: {elapsed:.2f}s"
    )


def category_image(category: str) -> Path:
    """Return a stable normal image; category routing, not defect detection, is under test."""
    image = DATASETS_ROOT / category / "test" / "good" / "000.png"
    assert image.is_file(), f"Required MVTec test image is missing: {image}"
    return image


@pytest.mark.parametrize("category", CATEGORIES, ids=CATEGORIES)
def test_category_patchcore_inference_and_response_schema(
    category: str,
    fast_patchcore_pipeline: dict[str, list[str]],
) -> None:
    """One parameterized test instance for every supported MVTec category."""
    response = category_router.route_inspection(str(category_image(category)), category)

    assert response["status"] == "success"
    assert RESPONSE_FIELDS.issubset(response)
    assert response["category"] == category
    assert response["prediction"] in {"Normal", "Anomaly"}
    assert isinstance(response["raw_score"], float)
    assert isinstance(response["normalized_score"], float)
    assert isinstance(response["threshold"], float)
    assert response["heatmap_path"] is not None
    assert Path(response["heatmap_path"]).is_file()
    assert fast_patchcore_pipeline["loaded_categories"] == [category]


def test_empty_image_is_rejected(tmp_path: Path) -> None:
    empty_image = tmp_path / "empty.png"
    empty_image.write_bytes(b"")

    with pytest.raises(ValueError):
        category_router.route_inspection(str(empty_image), "bottle")


def test_corrupt_image_is_rejected(tmp_path: Path) -> None:
    corrupt_image = tmp_path / "corrupt.png"
    corrupt_image.write_bytes(b"not a valid PNG image")

    with pytest.raises(ValueError):
        category_router.route_inspection(str(corrupt_image), "bottle")


def test_unsupported_category_returns_model_not_ready() -> None:
    response = category_router.route_inspection(str(category_image("bottle")), "unsupported")

    assert response == {
        "status": "model_not_ready",
        "category": "unsupported",
        "message": "No trained model available for 'unsupported'.",
    }


def test_missing_image_returns_error(tmp_path: Path) -> None:
    missing_image = tmp_path / "does_not_exist.png"

    response = category_router.route_inspection(str(missing_image), "bottle")

    assert response["status"] == "error"
    assert "Image not found" in response["message"]


def test_invalid_extension_is_rejected(tmp_path: Path) -> None:
    invalid_image = tmp_path / "invalid_extension.txt"
    invalid_image.write_text("this is not image data", encoding="utf-8")

    with pytest.raises(ValueError):
        category_router.route_inspection(str(invalid_image), "bottle")
