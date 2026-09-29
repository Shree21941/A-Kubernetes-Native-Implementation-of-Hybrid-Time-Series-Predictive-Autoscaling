from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_series  # noqa: E402


class LoadSeriesTests(unittest.TestCase):
    def test_fills_missing_minutes_for_period_count_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sparse.csv"
            pd.DataFrame({
                "period": ["1995-07-01 00:00", "1995-07-01 00:02"],
                "count": [4, 7],
            }).to_csv(path, index=False)

            series = load_series(path)

        self.assertEqual(series.tolist(), [4.0, 0.0, 7.0])
        self.assertTrue((series.index.to_series().diff().dropna()
                         == pd.Timedelta(minutes=1)).all())

    def test_accepts_processed_ds_y_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed.csv"
            pd.DataFrame({
                "ds": ["1995-07-01 00:00", "1995-07-01 00:02"],
                "y": [4, 7],
            }).to_csv(path, index=False)

            series = load_series(path)

        self.assertEqual(series.tolist(), [4.0, 0.0, 7.0])


if __name__ == "__main__":
    unittest.main()