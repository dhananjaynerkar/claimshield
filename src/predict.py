"""Score one raw insurance claim with the artifacts created by the notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def predict_claim(
    claim: Mapping[str, Any],
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    """Return a fraud score and review decision for one raw claim."""

    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / "fraud_ann_model.keras"
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    metadata_path = artifact_dir / "project_metadata.joblib"

    missing = [
        str(path)
        for path in (model_path, preprocessor_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Run the final notebook first: "
            + ", ".join(missing)
        )

    model = load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata = joblib.load(metadata_path)

    raw_claim = pd.DataFrame([dict(claim)])
    processed_claim = preprocessor.transform(raw_claim)
    probability = float(model.predict(processed_claim, verbose=0).ravel()[0])
    threshold = float(metadata["threshold"])
    prediction = int(probability >= threshold)

    return {
        "fraud_probability": round(probability, 6),
        "threshold": round(threshold, 6),
        "prediction": prediction,
        "decision": "FLAG_FOR_REVIEW" if prediction else "NO_FRAUD_FLAG",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        required=True,
        type=Path,
        help="Path to a JSON object containing one raw insurance claim.",
    )
    args = parser.parse_args()

    claim = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(claim, dict):
        raise ValueError("The input JSON must contain one object, not a list.")

    print(json.dumps(predict_claim(claim), indent=2))


if __name__ == "__main__":
    main()
