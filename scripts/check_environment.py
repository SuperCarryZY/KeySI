"""Check whether the local KeySI runtime has the expected files and imports."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


REQUIRED_MODULES = [
    "dash",
    "keybert",
    "matplotlib",
    "nltk",
    "numpy",
    "pandas",
    "plotly",
    "rank_bm25",
    "sklearn",
    "scipy",
    "sentence_transformers",
    "torch",
    "transformers",
]


def main() -> int:
    root = Path(os.getenv("KEYSI_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
    data_dir = Path(os.getenv("KEYSI_DATA_DIR", root / "CSV")).resolve()
    output_dir = Path(os.getenv("KEYSI_OUTPUT_DIR", root / "KeySI_results")).resolve()
    csv_path = data_dir / "risk_factors.csv"

    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]

    print(f"Project root: {root}")
    print(f"Data file:    {csv_path}")
    print(f"Output dir:   {output_dir}")

    if missing:
        print("\nMissing Python modules:")
        for name in missing:
            print(f"- {name}")
    else:
        print("\nPython imports: OK")

    if csv_path.exists():
        print("Input CSV: OK")
    else:
        print("Input CSV: missing")

    return 1 if missing or not csv_path.exists() else 0


if __name__ == "__main__":
    raise SystemExit(main())
