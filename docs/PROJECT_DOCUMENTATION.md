# ClaimShield Project Documentation

## Overview

ClaimShield is a supervised deep learning workflow for identifying vehicle insurance claims that may require fraud investigation. The model predicts a fraud probability and flags a claim when that probability meets the selected operating threshold. It is a decision-support model, not an automatic claim rejection system.

## Dataset

- Records: 15,420
- Columns: 33
- Target: `FraudFound_P` (`1` = fraud, `0` = non-fraud)
- Fraud cases: 923
- Fraud rate: 5.99%
- Source file: `data/raw/FraudDataset.csv`

The local source folder did not contain upstream provenance or a redistribution license. Confirm that the dataset may be published before making the CSV public. If redistribution is not permitted, keep the CSV ignored and document how an authorized user should provide it locally.

## Workflow

1. Load and inspect the raw claim data.
2. Check missing values, duplicate rows, skewness, outliers, class distribution, and categorical fraud rates.
3. Drop `PolicyNumber`, `RepNumber`, and `Year` from the model features.
4. Replace invalid zero values in age and claim date fields with training-safe dataset values.
5. Split the data into stratified train, validation, and test sets using a 64% / 16% / 20% split.
6. Fit a `ColumnTransformer` on the training data only.
7. Apply `StandardScaler` to numeric and ordinal features and one-hot encoding to nominal features.
8. Apply SMOTE only to the processed training data and retain class weights for fraud-sensitive training.
9. Train a baseline ANN and a regularized ANN with Batch Normalization, Dropout, L2 regularization, Adam, and early stopping.
10. Select the probability threshold using validation-set F2 score.
11. Evaluate the selected model once on the untouched test set.
12. Save the model, preprocessing pipeline, metadata, and threshold for reloadable inference.

## ANN Architecture

The tuned network uses 102 processed input features, Dense layers of 64 and 32 units with ReLU activation, Batch Normalization, 40% Dropout, L2 regularization, and a sigmoid output for binary classification.

## Reproducible Validation Run

The final notebook was executed successfully in the project environment with Python 3.11 and TensorFlow 2.21.0 on CPU. The following values are from that run and are not hard-coded summaries:

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
| True negatives | 1,775 |
| False positives | 1,124 |
| False negatives | 22 |
| True positives | 163 |

The low precision means the model sends many legitimate claims for manual review. This is expected from a recall-focused threshold and must be considered when choosing an operational review capacity.

## Generated Artifacts

Running the notebook creates these files under `artifacts/`:

- `fraud_ann_model.keras`: final trained ANN.
- `best_ann_model.keras`: best validation-AUC checkpoint.
- `preprocessor.joblib`: fitted numeric, ordinal, and nominal feature transformations.
- `project_metadata.joblib`: feature lists, encoding configuration, threshold, SMOTE ratio, and class weights.

These files are intentionally ignored by Git and can be recreated from the notebook and dataset.

## Inference

After running the notebook, `src/predict.py` can load the saved artifacts and score one claim represented as a JSON object. The object should contain the same raw feature columns used by the notebook. Extra columns are ignored by the saved preprocessor; missing required columns will raise a clear error.

Example:

```powershell
.\.venv\Scripts\python.exe src\predict.py --input-json claim.json
```

The output contains the fraud probability, selected threshold, binary prediction, and a review decision. The artifacts must exist before this command is run.

## Limitations and Future Work

- Precision is low at the recall-focused operating point; threshold selection should use confirmed business costs and investigator capacity.
- The dataset is historical and relatively small for deep learning; performance may not generalize to a new insurer or time period.
- Probability calibration, temporal validation, drift monitoring, and explainability should be added before production use.
- A controlled API or dashboard can be added around the artifact-driven inference function.
- Data provenance and redistribution permission must be recorded before public release of the CSV.
