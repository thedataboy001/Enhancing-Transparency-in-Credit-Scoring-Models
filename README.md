# Enhancing Transparency in Credit Scoring Models

This repository contains a project that explores explainable AI techniques for loan approval and credit scoring models. It is a refactor and rework of a dissertation titled "Enhancing Transparency in Credit Scoring Models: An Explainable AI Techniques on Loan Approval System". It includes preprocessing, exploratory data analysis, model training, and explainability notebooks.

Project background:
- This repository started as a dissertation project. The first implementation was developed in a single notebook titled `Enhancing Transparency in Credit Scoring Models - Exploring Explainable AI Techniques for Loan Approval Systems- Normalised Data 3.ipynb` (the "original/legacy" notebook).
- The original notebook has been refactored and reworked: core logic has been structured into reusable functions and scripts like `preprocess.py`, `data_utils.py`, and `scaler.py`. The `01_*`, `02_*`, and `03_*` notebooks are the recommended workflow and demonstrate modular usage.
- The legacy notebook is retained for reference but the modularized code is the recommended codebase for testing and deployment.

Structure:
- `data/` - raw data files (`labelled_data.csv`, `unlabelled_data.csv`).
- `01_data_preprocessing.ipynb` - Data preprocessing notebook.
- `02_data_analysis.ipynb` - EDA and model training (was `data_analysis.ipynb`).
- `03_xai_loan_approval_normalised_v3.ipynb` - Notebook focused on XAI techniques (was the long-named notebook).
- `preprocess.py` - Preprocessing functions and small CLI entrypoint (was `helper.py`).
- `data_utils.py` - Utility functions for data parsing and conversion (was `helper_functions.py`).
- `scaler.py` - Preprocessing pipeline for transformers.
- `data_dictionary.md` - Data dictionary.

How to use:
1. Make sure dependencies are installed (see `requirements.txt` if present). Typical packages include numpy, pandas, scikit-learn, seaborn, matplotlib, xgboost.
2. Run `01_data_preprocessing.ipynb` to perform cleaning and save processed data if needed (or run `preprocess.py`). The preprocessing logic is also available via `preprocess.py` for programmatic or CI-based workflows.
3. Run `02_data_analysis.ipynb` to train models and explore evaluations.
4. Run `03_xai_loan_approval_normalised_v3.ipynb` for explainability techniques — this runs the original long-named notebook if you prefer the legacy content.

Notes:
- Files were renamed to be consistent and easier to navigate: snake_case, short but descriptive names.
- Both the original notebooks (`data.ipynb`, `data_analysis.ipynb`, and the long-named `Enhancing Transparency...Normalised Data 3.ipynb`) remain in the repo as legacy copies; prefer the numbered notebooks (`01_*`, `02_*`, `03_*`) for workflow ordering.
- To update code referencing old filenames, search for `helper`, `helper_functions`, or old notebook names and update them accordingly.
 
Note on backwards compatibility:
- `preprocess.py` and `data_utils.py` are the canonical modules; we recommend you import from these. Some legacy code or notebooks may still reference the older notebook files — those are retained to avoid accidental breakage.
Deployment-ready notes:
- The repository now includes modular Python files (`preprocess.py`, `data_utils.py`, `scaler.py`) that are suitable for reuse in scripts or web applications. To prepare this project for actual production deployment, consider adding a `setup.py` / `pyproject.toml`, packaging the preprocessing and modeling code into a module, and adding API endpoints or CLI wrappers.
File mapping (old → new):

- `data.ipynb` → `01_data_preprocessing.ipynb` (preprocessing notebook)
- `data_analysis.ipynb` → `02_data_analysis.ipynb` (EDA & training)
- `Enhancing Transparency...Normalised Data 3.ipynb` → `03_xai_loan_approval_normalised_v3.ipynb` (short wrapper created; the original is kept)
- `helper.py` → `preprocess.py` (moved and improved)
- `helper_functions.py` → `data_utils.py` (moved and improved)
- `note` → `data_dictionary.md`

Quick start (local):

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv;
.venv\Scripts\Activate.ps1
```

2. Install dependencies (if you have a `requirements.txt`) or at least the typical packages used in the notebooks:

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn xgboost notebook
```

3. Run the preprocessing notebook or script:

```powershell
jupyter notebook 01_data_preprocessing.ipynb
# Or run the script term:
python preprocess.py
```

4. Run the analysis notebook:

```powershell
jupyter notebook 02_data_analysis.ipynb
```

Optional cleanup:
- If you prefer a consolidated folder, you can remove or archive the older notebooks (`data.ipynb`, `data_analysis.ipynb`, and the long-named notebook) once you have verified the new files work as expected.

If this project is intended as a submission or code artifact for an academic assessment, I recommend keeping the legacy notebook and the refactored content in the repo (with clear README notes) so reviewers can see the original work and the improved, deployable implementation.

