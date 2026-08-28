# ClaimShield reproducibility notes

## Repository snapshot

- Repository: dhananjaynerkar/claimshield
- Documented workflow commit: 5cd5e54a08fd5e1b092b8a38b6fb9e2620760eb8
- Verified runtime described by the project: Python 3.11, TensorFlow 2.21.0, CPU

## Environment

Install the pinned dependencies from requirements.txt:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

The requirements file records the package versions used by the documented run. A clean run should capture the Python version, operating system, commit SHA, and whether the run completed on CPU or another device.

## Input data gate

The notebook expects data/raw/FraudDataset.csv. Its redistribution rights are not verified. Do not download, commit, or redistribute a replacement input until the provenance gate in DATASET_PROVENANCE.md is closed.

## Training and artifacts

Run the notebook:

~~~powershell
.\.venv\Scripts\jupyter-notebook.exe
~~~

The final notebook creates the ignored artifacts needed by src/predict.py:

- artifacts/fraud_ann_model.keras
- artifacts/preprocessor.joblib
- artifacts/project_metadata.joblib

## Inference

After the artifacts exist, score one JSON claim:

~~~powershell
.\.venv\Scripts\python.exe src\predict.py --input-json claim.json
~~~

The command returns a fraud probability, selected threshold, binary prediction, and review decision. Missing artifacts should fail clearly rather than silently generating a result.

## Reproduction record

For every future run, record:

- commit SHA;
- dataset source/version and permission status;
- Python and dependency versions;
- split and threshold-selection protocol;
- model artifact names and checksums;
- test metrics and confusion matrix;
- any warnings, failures, or skipped steps.

