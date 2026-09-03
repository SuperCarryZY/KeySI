# Reproducibility

## Input Contract

The default input is `CSV/risk_factors.csv`.

Expected schema:

- column 1: document label or group identifier;
- column 2: article/document text.

Rows with missing text are skipped by several preprocessing paths. Very short documents may also be excluded during keyword extraction.

## Environment

Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

Run the environment check:

```powershell
python scripts/check_environment.py
```

## Execution Protocol

```powershell
$env:PYTHONPATH="src"
python -m keysi_app
```

Open:

```text
http://127.0.0.1:47983
```

The application writes generated outputs to `KeySI_results/`.

By default, user refinement history is preserved across restarts. Set `KEYSI_RESET_USER_DATA_ON_START=1` only when intentionally starting a fresh session.

## Artifact Checklist

Before releasing with a paper, record:

- commit hash;
- input dataset checksum;
- Python version;
- package versions;
- GPU/CPU hardware;
- random seed;
- generated model checkpoints;
- generated metrics JSON files;
- any manual user-refinement actions used in reported results.
