"""Train and persist the online exam pass prediction model.

Run from the repository root:
    python scripts/train_model.py

This produces models/logistic_regression.pkl, matching the pipeline used by app.py.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "online_exam_data.csv"
MODEL_FILE = ROOT / "models" / "logistic_regression.pkl"
FEATURES = [
    "study_hours",
    "previous_score",
    "attendance",
    "assignment_score",
    "practice_tests",
    "online_participation",
    "sleep_hours",
]
TARGET = "exam_result"


def main() -> None:
    data = pd.read_csv(DATA_FILE)
    for column in FEATURES:
        data[column] = data[column].fillna(data[column].median())
    data[TARGET] = data[TARGET].fillna(data[TARGET].mode()[0])

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )
    model.fit(data[FEATURES], data[TARGET])

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Saved trained model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
