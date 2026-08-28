from pathlib import Path

import numpy as np

from src import predict


class FakePreprocessor:
    def transform(self, raw_claim):
        assert len(raw_claim) == 1
        return raw_claim


class FakeModel:
    def predict(self, processed_claim, verbose=0):
        assert len(processed_claim) == 1
        assert verbose == 0
        return np.array([[0.73]])


def test_predict_claim_returns_stable_artifact_contract(monkeypatch, tmp_path: Path):
    for filename in (
        "fraud_ann_model.keras",
        "preprocessor.joblib",
        "project_metadata.joblib",
    ):
        (tmp_path / filename).write_text("fixture", encoding="utf-8")

    monkeypatch.setattr(predict, "load_model", lambda _path: FakeModel())
    monkeypatch.setattr(
        predict.joblib,
        "load",
        lambda path: (
            FakePreprocessor()
            if path.name == "preprocessor.joblib"
            else {"threshold": 0.30}
        ),
    )

    result = predict.predict_claim({"claim_id": "fixture-1"}, artifact_dir=tmp_path)

    assert result == {
        "fraud_probability": 0.73,
        "threshold": 0.3,
        "prediction": 1,
        "decision": "FLAG_FOR_REVIEW",
    }

