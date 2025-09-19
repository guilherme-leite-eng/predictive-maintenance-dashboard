# src/train.py
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

SEED = 42

def build_pipeline(cat_cols, numeric_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="passthrough",
    )
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    return pipe

def train(data_path, out_dir):
    np.random.seed(SEED)

    df = pd.read_csv(data_path)
    target_col = "Target"
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not in dataset columns")

    # Drop obvious identifiers if present
    drop_candidates = ["UDI", "UID", "Failure Type"]
    drop_cols = [c for c in drop_candidates if c in df.columns]
    X = df.drop(columns=drop_cols + [target_col])
    y = df[target_col].astype(int)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pipe = build_pipeline(cat_cols, numeric_cols)
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("ROC AUC:", auc)
    print("Classification report keys:", list(report.keys()))

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "pm_pipeline.joblib")
    joblib.dump(pipe, model_path)

    # background sample (for SHAP)
    bg_sample = X_train.sample(min(200, len(X_train)), random_state=SEED)
    bg_sample.to_csv(os.path.join(out_dir, "background.csv"), index=False)

    # feature names after preprocessing (for mapping importance)
    ohe_names = []
    try:
        ohe = pipe.named_steps["pre"].named_transformers_["ohe"]
        if hasattr(ohe, "get_feature_names_out"):
            ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    except Exception:
        ohe_names = []

    feature_names = ohe_names + numeric_cols
    with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    metrics = {"roc_auc": auc, "classification_report": report, "confusion_matrix": cm}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model and artifacts to {out_dir}")
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to predictive_maintenance.csv")
    parser.add_argument("--out", default="artifacts", help="Output artifacts directory")
    args = parser.parse_args()
    train(args.data, args.out)
