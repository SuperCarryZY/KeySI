# KeySI

KeySI is an interactive research prototype for keyword-guided semantic inspection, clustering, triplet/prototype training, and user refinement over article text.

This repository is organized as a publication-ready code package: the original prototype is preserved as the executable core, while documentation, configuration, environment setup, and reproducibility notes are separated from the application logic.

## Repository Layout

```text
.
├── configs/default.yaml          # Paper-facing configuration summary
├── docs/ARCHITECTURE.md          # System and module-boundary notes
├── docs/REPRODUCIBILITY.md       # Data, environment, and run protocol
├── scripts/check_environment.py  # Lightweight setup validation
├── src/keysi_app/                # Importable Python package
│   ├── __main__.py               # python -m keysi_app entry point
│   └── keysiformal.py            # Preserved prototype implementation
└── tests/                        # Place focused regression tests here
```

Legacy root-level prototype files from the original repository are retained for traceability. New work should prefer the package entry point under `src/keysi_app/`.

## Data

Place the input file at:

```text
CSV/risk_factors.csv
```

The current implementation expects the first column to contain labels and the second column to contain article text.

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

## Publication Notes

For conference submission, report the exact dataset version, model checkpoint policy, random seeds, hardware, dependency versions, and output artifacts. The current prototype produces artifacts under `KeySI_results/`; keep those generated files out of source control unless the paper explicitly requires releasing fixed outputs.
