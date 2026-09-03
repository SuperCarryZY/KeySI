# Architecture

The original KeySI prototype mixed application state, preprocessing, model definition, training, evaluation, visualization, and Dash callbacks in one file. The current repository layout separates the project shell from the preserved implementation so the code package can be reviewed and executed consistently.

## Current Runtime

`src/keysi_app/keysiformal.py` remains the behavior-preserving implementation. It now resolves project paths from the repository root by default and supports environment-variable overrides:

- `KEYSI_PROJECT_ROOT`
- `KEYSI_DATA_DIR`
- `KEYSI_OUTPUT_DIR`
- `KEYSI_HOST`
- `KEYSI_PORT`
- `KEYSI_DEBUG`
- `KEYSI_RANDOM_SEED`
- `KEYSI_RESET_USER_DATA_ON_START`

## Recommended Module Boundaries

The next refactor should split the preserved implementation along these boundaries:

- `config.py`: path resolution, constants, and training hyperparameters.
- `data.py`: CSV loading, valid-document indexing, BM25 caches, and user-data persistence.
- `models.py`: `SentenceEncoder`, checkpoint validation, tokenizer/model loading.
- `keywords.py`: NLTK preprocessing, KeyBERT extraction, keyword matching.
- `training.py`: triplet/prototype training, snapshot creation, clustering metrics.
- `visualization.py`: t-SNE generation, Plotly figure construction, highlighting helpers.
- `callbacks.py`: Dash callback registration only.
- `layout.py`: Dash layout and reusable UI components.
- `app.py`: app factory and command-line entry point.

That split should be done with regression checks after each boundary move because many callbacks currently share mutable global state.

## Conference-Code Expectations

For a top-conference artifact, the code should make these concerns explicit:

- deterministic run settings and random seeds;
- exact model and dependency versions;
- dataset schema and preprocessing decisions;
- trained artifact names and regeneration steps;
- evaluation script independent from the interactive UI;
- ablation or sensitivity settings separated from default demo settings.
