import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path(__file__).resolve().parent / "data" / "parkinsons.data"
ARTIFACT_PATH = Path(__file__).resolve().parent / "parkinsons_model.joblib"

DATASET_FEATURE_COLUMNS = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]

EXTRACTED_TO_DATASET_FEATURES = {
    "MDVP_Fo_Hz": "MDVP:Fo(Hz)",
    "MDVP_Fhi_Hz": "MDVP:Fhi(Hz)",
    "MDVP_Flo_Hz": "MDVP:Flo(Hz)",
    "MDVP_Jitter_pct": "MDVP:Jitter(%)",
    "MDVP_Jitter_Abs": "MDVP:Jitter(Abs)",
    "MDVP_RAP": "MDVP:RAP",
    "MDVP_PPQ": "MDVP:PPQ",
    "Jitter_DDP": "Jitter:DDP",
    "MDVP_Shimmer": "MDVP:Shimmer",
    "MDVP_Shimmer_dB": "MDVP:Shimmer(dB)",
    "Shimmer_APQ3": "Shimmer:APQ3",
    "Shimmer_APQ5": "Shimmer:APQ5",
    "MDVP_APQ": "MDVP:APQ",
    "Shimmer_DDA": "Shimmer:DDA",
    "NHR": "NHR",
    "HNR": "HNR",
    "RPDE": "RPDE",
    "DFA": "DFA",
    "spread1": "spread1",
    "spread2": "spread2",
    "D2": "D2",
    "PPE": "PPE",
}

RISK_BANDS = [
    {
        "label": "Low Likelihood",
        "color": "#00C896",
        "description": "The trained model found a low likelihood of Parkinsonian voice patterns in this sample.",
        "recommendation": "If symptoms are present, repeat the recording in a quiet room and discuss the result with a clinician.",
    },
    {
        "label": "Mild Risk",
        "color": "#FFD166",
        "description": "The model detected mild voice irregularities associated with Parkinsonian speech patterns.",
        "recommendation": "Consider a repeat screening and clinical follow-up if voice or motor symptoms are noticeable.",
    },
    {
        "label": "Elevated Risk",
        "color": "#FF9F43",
        "description": "The model detected a stronger cluster of vocal markers often seen in Parkinson's disease samples.",
        "recommendation": "A neurological evaluation is recommended, especially if this matches other symptoms.",
    },
    {
        "label": "High Risk",
        "color": "#EE4B6A",
        "description": "The model found a high likelihood of Parkinsonian voice patterns in this recording.",
        "recommendation": "Seek formal medical evaluation promptly. This tool is only a screening aid and not a diagnosis.",
    },
]


def _load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    rows = []
    labels = []
    with DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([float(row[column]) for column in DATASET_FEATURE_COLUMNS])
            labels.append(int(row["status"]))

    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _candidate_models():
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
    }


def train_and_save_model(force: bool = False):
    if ARTIFACT_PATH.exists() and not force:
        return joblib.load(ARTIFACT_PATH)

    features, labels = _load_dataset()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_name = None
    best_cv_score = -np.inf
    best_model = None
    cv_scores = {}

    for model_name, model in _candidate_models().items():
        scores = cross_val_score(model, features, labels, cv=cv, scoring="roc_auc")
        cv_scores[model_name] = float(np.mean(scores))
        if cv_scores[model_name] > best_cv_score:
            best_cv_score = cv_scores[model_name]
            best_name = model_name
            best_model = model

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    best_model.fit(X_train, y_train)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    predictions = best_model.predict(X_test)

    artifact = {
        "model": best_model,
        "feature_columns": DATASET_FEATURE_COLUMNS,
        "feature_mapping": EXTRACTED_TO_DATASET_FEATURES,
        "model_name": best_name,
        "metrics": {
            "cv_roc_auc": round(best_cv_score, 4),
            "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "test_f1": round(float(f1_score(y_test, predictions)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "class_balance": {
                "healthy": int(np.sum(labels == 0)),
                "parkinsons": int(np.sum(labels == 1)),
            },
            "candidates": {name: round(score, 4) for name, score in cv_scores.items()},
        },
        "risk_bands": RISK_BANDS,
    }

    joblib.dump(artifact, ARTIFACT_PATH)
    return artifact


def load_model_artifact():
    if ARTIFACT_PATH.exists():
        return joblib.load(ARTIFACT_PATH)
    return train_and_save_model()


def build_model_vector(extracted_features: dict, feature_mapping: dict):
    dataset_features = {}
    for extracted_name, dataset_name in feature_mapping.items():
        if extracted_name not in extracted_features:
            raise KeyError(f"Missing extracted feature: {extracted_name}")
        dataset_features[dataset_name] = float(extracted_features[extracted_name])

    ordered_values = [dataset_features[column] for column in DATASET_FEATURE_COLUMNS]
    return np.asarray(ordered_values, dtype=np.float64).reshape(1, -1)


def severity_from_probability(probability: float):
    if probability < 0.35:
        return 0
    if probability < 0.55:
        return 1
    if probability < 0.75:
        return 2
    return 3
