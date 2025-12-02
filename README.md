# Enhancing Transparency in Credit Scoring Models

This repository contains a refactored implementation of a dissertation project titled "Enhancing Transparency in Credit Scoring Models: An Explainable AI Techniques on Loan Approval System". The codebase centralises preprocessing, modeling, and explainability logic into short, reusable modules and notebooks.

Why this repo exists
- The original dissertation code was created as a single notebook (legacy/first implementation: `Enhancing Transparency in Credit Scoring Models - Exploring Explainable AI Techniques for Loan Approval Systems- Normalised Data 3.ipynb`).
- The project has been refactored into modular scripts and notebooks for reusability, reproducibility, and easy deployment.

Key project goals
- Make credit scoring model development transparent by adding explainability tools (SHAP, DiCE, etc.)
- Separate preprocessing pipelines and training routines so they are reusable and testable
- Make it easy to reproduce, run and extend the pipeline and produce consistent outputs.

Dataset
- We use the Kaggle dataset "Credit Score Classification" (parisrohan). The CSVs used in this repo are in `data/` as `labelled_data.csv` and `unlabelled_data.csv`. Please follow the dataset license and citation rules when distributing or publishing the dataset.

Repository structure (high level)
- `data/` — raw CSVs (labelled/unlabelled)
- `data_analysis.ipynb` — interactive exploration, feature engineering and model experiments
- `preprocess.py` — small helpers for preprocessing functions (module)
- `data_utils.py` — reusable utilities (age parsing, helpers)
- `preprocess_steps.py` — standalone preprocessing pipeline (full run -> writes processed CSV)
- `utils_and_constants.py` — central path/column constants used by multiple scripts
- `train_model.py` — orchestrates training for DT, RF, and XGBoost, saves metrics and confusion matrices
- `dt_model.py`, `rf_model.py`, `xgb_model.py` — training/eval functions for each model
- `metrics_and_plots.py` — helpers used by `train_model.py` to save metrics/plots
- `processed_dataset/` — generated output (the repo creates it when required)
- `data_dictionary.md` — column definitions and data dictionary

Quick start (local)
1) Create and activate a venv (recommended):
```powershell
python -m venv .venv;
.venv\Scripts\Activate.ps1
```
2) Install dependencies:
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn xgboost notebook
```
3) Preprocess and produce the processed dataset: (this will create `processed_dataset/transformed_data.csv`)
```powershell
python preprocess_steps.py
```
4) Train models and save metrics:
```powershell
python train_model.py
```

CLI-style, dependency-driven examples (informational)
- These examples highlight which files are used by a script. The scripts do not currently support these flags — this is intended as a human-friendly representation of dependencies.

Preprocess pipeline (informational):
```powershell
# Depends on: utils_and_constants.py, data/labelled_data.csv, preprocess_steps.py
# Output: processed_dataset/transformed_data.csv
python preprocess_steps.py
```

Train (informational):
```powershell
# Depends on: utils_and_constants.py, processed_dataset/transformed_data.csv, rf_model.py, xgb_model.py, dt_model.py, metrics_and_plots.py
python train_model.py
```

Expected outputs
- `processed_dataset/transformed_data.csv` — processed dataset created by preprocessing pipeline
- `rf_model_metrics.json`, `xgb_model_metrics.json`, `dt_model_metrics.json` — evaluation metrics saved by `train_model.py`
- `rf_model_confusion_matrix.png`, `xgb_model_confusion_matrix.png`, `dt_model_confusion_matrix.png` — confusion matrix figures saved by `train_model.py`
- Optional: `counterfactuals.csv` produced by running XAI steps in `03_xai_loan_approval_normalised_v3.ipynb`

Recommended pipeline (interactive or CLI)
1) Confirm the raw data exists in `data/`.
2) Run `preprocess_steps.py` to generate `processed_dataset/transformed_data.csv`.
3) Run `train_model.py` to train models and generate metrics + plots.
4) Optionally run the EDA notebook (`data_analysis.ipynb`) for more analysis and to generate comparison plots.

Notes and troubleshooting
- If `processed_dataset/` does not exist, `preprocess_steps.py` will create it. Make sure you have write access in the repository root.
- Scripts use constants in `utils_and_constants.py` to set file paths and column lists. Use that file to update constants if you reorganize your folders.
- If you want a strict command-line interface for dependencies and custom paths, consider adding argument parsing (argparse) to `preprocess_steps.py` and `train_model.py`.

Contributing & reproducibility
- If you plan to contribute: create an issue or PR with your proposed changes. Add unit tests for core modules and a `requirements.txt` or `pyproject.toml` for reproducibility.
- To create a requirements file locally for this environment you can run:
```powershell
pip freeze > requirements.txt
```

License
- No license included in this repository. If you plan to distribute, add a license like MIT or Apache-2.0.

Academic note
- The repository is a rework of my original dissertation work; both the original notebook and the structured rework are kept for traceability. The reworked code is structured to support reproducible flows and easier deployment.
