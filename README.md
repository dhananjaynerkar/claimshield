# ClaimShield - Insurance Fraud Detection using ANN

ClaimShield is a deep learning workflow for detecting vehicle insurance claims that may require fraud investigation. It combines leakage-aware preprocessing, imbalance handling, a regularized Artificial Neural Network, validation-based threshold tuning, and reloadable inference artifacts.

## Highlights

- 15,420 insurance claim records with a 5.99% fraud rate.
- Stratified train, validation, and test split.
- Numeric scaling, ordinal encoding, and one-hot encoding through `ColumnTransformer`.
- SMOTE applied only to the training set, with fraud-sensitive class weights.
- ANN with ReLU, Batch Normalization, Dropout, L2 regularization, Adam, and early stopping.
- Threshold selected on validation F2 score; final metrics calculated on untouched test data.
- Reusable model, preprocessor, metadata, and threshold artifacts.

## Project Structure

```text
claimshield/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/
│       ├── README.md
│       └── FraudDataset.csv
├── docs/
│   └── PROJECT_DOCUMENTATION.md
├── notebooks/
│   └── Insurance_Fraud_Detection_ANN_Clean_DL_Engineer.ipynb
├── artifacts/
│   └── README.md
└── src/
    └── predict.py
```

## Setup on Windows

Use Python 3.11, which is the version used for the verified run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open the notebook with the project environment:

```powershell
.\.venv\Scripts\jupyter-notebook.exe
```

Then open `notebooks/Insurance_Fraud_Detection_ANN_Clean_DL_Engineer.ipynb` and run all cells. The notebook discovers the project root from the dataset location, so it can be opened from the project root or the `notebooks/` folder.

## Validation Results

These values are from a complete notebook execution in Python 3.11 with TensorFlow 2.21.0 on CPU:

| Metric | Test result |
| --- | ---: |
| Selected threshold | 0.30 |
| Accuracy | 0.6284 |
| Fraud precision | 0.1267 |
| Fraud recall | 0.8811 |
| F1 score | 0.2215 |
| F2 score | 0.4021 |
| ROC-AUC | 0.8161 |
| Average precision / PR-AUC | 0.1829 |
| Confusion matrix | TN=1,775; FP=1,124; FN=22; TP=163 |

The threshold prioritizes fraud recall and therefore creates false positives for manual review. The model should support investigation, not automatically reject claims.

## Inference

Run the notebook first to create the ignored files in `artifacts/`. Then score one JSON claim:

```powershell
.\.venv\Scripts\python.exe src\predict.py --input-json claim.json
```

The JSON object should contain the raw claim columns used by the notebook. The command returns the fraud probability, selected threshold, prediction, and review decision.

## Dataset Note

The repository contains `data/raw/FraudDataset.csv` because it is required for notebook execution. The local source folder did not include upstream provenance or a redistribution license. Verify publication rights before making this repository public; if redistribution is not permitted, keep the CSV local and provide an authorized download or placement instruction.

## Project documents

- [Model card](docs/MODEL_CARD.md) — intended use, training workflow, evaluation checkpoint, and limitations.
- [Reproducibility notes](docs/REPRODUCIBILITY.md) — environment, artifact contract, and run-record fields.
- [Dataset provenance](docs/DATASET_PROVENANCE.md) — publication-rights gate for the raw CSV.

## Technology Stack

Python, TensorFlow, Keras, scikit-learn, imbalanced-learn, pandas, NumPy, Matplotlib, Seaborn, Joblib, and Jupyter Notebook.

See [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) for the complete workflow, evaluation interpretation, artifact details, limitations, and future improvements.
