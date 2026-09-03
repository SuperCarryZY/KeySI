# KeySI

KeySI is an interactive system for keyword-guided semantic inspection, clustering, triplet/prototype training, and user refinement over article text.

This repository contains the code and data used for the accepted KeySI paper. The original runnable scripts are preserved for traceability, and the organized package under `src/keysi_app/` provides the preferred entry point.

## Repository Layout

```text
.
├── 20news_6class_cleaned.csv     # 20 Newsgroups six-class dataset
├── CSV/risk_factors.csv          # Risk factor dataset path used by the app
├── configs/default.yaml          # Default configuration summary
├── docs/ARCHITECTURE.md          # System and module-boundary notes
├── docs/REPRODUCIBILITY.md       # Data, environment, and run protocol
├── scripts/check_environment.py  # Lightweight setup validation
├── src/keysi_app/                # Importable Python package
│   ├── __main__.py               # python -m keysi_app entry point
│   └── keysiformal.py            # Preserved implementation
└── tests/                        # Place focused regression tests here
```

Legacy root-level files are retained for traceability. New work should prefer the package entry point under `src/keysi_app/`.

## Data

The project uses two datasets:

- `20news_6class_cleaned.csv`: cleaned six-class 20 Newsgroups dataset.
- `CSV/risk_factors.csv`: risk factor dataset used by the app runtime.

The app runtime expects the active dataset at:

```text
CSV/risk_factors.csv
```

The current implementation expects the first column to contain labels and the second column to contain article text. To run a different CSV with the app, either place it at `CSV/risk_factors.csv` or point `KEYSI_DATA_DIR` to a folder containing `risk_factors.csv`.

You can override paths without editing code:

```powershell
$env:KEYSI_DATA_DIR="C:\path\to\CSV"
$env:KEYSI_OUTPUT_DIR="C:\path\to\KeySI_results"
```

## Run

```powershell
python -m pip install -r requirements.txt
python scripts/check_environment.py
$env:PYTHONPATH="src"
python -m keysi_app
```

Default local URL:

```text
http://127.0.0.1:47983
```

Optional runtime overrides:

```powershell
$env:KEYSI_HOST="127.0.0.1"
$env:KEYSI_PORT="47983"
$env:KEYSI_DEBUG="1"
$env:KEYSI_RANDOM_SEED="42"
$env:KEYSI_RESET_USER_DATA_ON_START="0"
```

## Main Workflow

1. Choose the number of concept groups.
2. Click **Generate Groups**.
3. Select a group, then click keywords in **Keywords 2D Visualization** to assign them.
4. Use **Exclude** for irrelevant or noisy keywords.
5. Click **Train Model** and inspect the before/after projections in Training View.
6. Use Refinement Mode to move documents between groups or exclude documents, then rerun refinement training.

## Notes

Generated outputs are written to `KeySI_results/`. These files are ignored by git because they include local runs, checkpoints, plots, and user-refinement histories.
