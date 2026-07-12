"""
PatchCore Score Scaler
======================

Loads category-specific MinMax scalers and
normalizes PatchCore anomaly scores to 0-100.

Author: Srinath
"""

from pathlib import Path
import joblib


class ScoreScaler:

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[2]

        self.scaler_dir = self.project_root / "models" / "scalers"

        self.cache = {}

    def load(self, category):

        if category not in self.cache:

            scaler_path = self.scaler_dir / f"{category}_scaler.pkl"

            if not scaler_path.exists():
                raise FileNotFoundError(
                    f"Scaler not found:\n{scaler_path}"
                )

            self.cache[category] = joblib.load(scaler_path)

        return self.cache[category]

    def normalize(self, score, category):

        scaler = self.load(category)

        value = scaler.transform([[score]])[0][0]

        value = max(0, min(100, value))

        return float(value)