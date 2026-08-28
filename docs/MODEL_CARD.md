# ClaimShield model card

## Model purpose

ClaimShield is a decision-support workflow for flagging vehicle insurance claims that may require fraud investigation. It is not an automatic claim-rejection system.

## Data and target

- Records: 15,420
- Columns: 33
- Target: FraudFound_P, where 1 indicates fraud and 0 indicates non-fraud
- Fraud cases: 923
- Fraud rate: 5.99%
- Split: stratified 64% train, 16% validation, 20% untouched test

The dataset publication gate is unresolved. See DATASET_PROVENANCE.md before redistributing the raw CSV.

## Training workflow

1. Remove PolicyNumber, RepNumber, and Year from model features.
2. Fit the ColumnTransformer on training data only.
3. Scale numeric and ordinal features and one-hot encode nominal features.
4. Apply SMOTE only to processed training data and retain fraud-sensitive class weights.
5. Train a regularized Artificial Neural Network with Dense layers, ReLU activations, Batch Normalization, Dropout, L2 regularization, Adam, and early stopping.
6. Select the operating threshold using validation-set F2 score.
7. Evaluate once on the untouched test set.
8. Save the model, preprocessing pipeline, metadata, and threshold for reloadable inference.

## Documented evaluation checkpoint

The README records a complete notebook execution using Python 3.11 and TensorFlow 2.21.0 on CPU. The repository snapshot containing the documented workflow is commit 5cd5e54a08fd5e1b092b8a38b6fb9e2620760eb8.

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

## Intended use

Use the output to prioritize claims for human investigation and to inspect the trade-off between fraud recall and manual-review volume.

## Limitations

- The selected threshold produces many false positives; operational use requires an approved investigator capacity and cost model.
- The data is historical and may not represent a new insurer, product, or time period.
- Calibration, temporal validation, drift monitoring, and external validation are not established by this checkpoint.
- Performance should not be presented as production impact or automatic fraud adjudication.

