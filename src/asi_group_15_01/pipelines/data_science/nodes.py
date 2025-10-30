import pandas as pd
import numpy as np
import wandb
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def load_raw(raw_csv: str) -> pd.DataFrame:
    """Load raw data from a CSV file.

    Args:
        raw_csv (str): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Raw data.
    """
    return pd.read_csv(raw_csv, skipinitialspace=True)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning on the raw data.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    # Remove duplicates if any
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)

    # Replace '?' with NaN and drop rows with NaN values
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Drop unnecessary columns
    df.drop(columns=["fnlwgt", "education"], inplace=True)

    # Encode categorical variables
    columns_to_encode = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    df = pd.get_dummies(
        df,
        columns=columns_to_encode,
        prefix=columns_to_encode,
        drop_first=True,
    )

    # Log-transform the 'capital-gain' and 'capital-loss' columns
    df["capital-gain-log"] = np.log1p(df["capital-gain"])
    df["capital-loss-log"] = np.log1p(df["capital-loss"])
    df = df.drop(columns=["capital-gain", "capital-loss"])

    # Convert target variable to binary
    mapping = {"<=50K": 0, ">50K": 1}
    df["income_encoded"] = df["income"].map(mapping)
    df.drop(columns=["income"], inplace=True)

    return df


def train_test_split(
    df: pd.DataFrame, split_params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the cleaned data into training and testing sets using parameters from config."""
    y = df["income_encoded"]
    X = df.drop(columns=["income_encoded"])

    X_train, X_test, y_train, y_test = sk_train_test_split(
        X,
        y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=y if split_params.get("stratify", True) else None,
    )

    y_train = y_train.to_frame()
    y_test = y_test.to_frame()

    return X_train, X_test, y_train, y_test


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series, model_params: dict):
    """Train a model with parameters from config."""

    run = wandb.init(
        project="asi-group-15-01",
        job_type="train",
        config=model_params,
        reinit=True,
    )

    if model_params["kind"] == "random_forest":
        model = RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth"),
            random_state=model_params.get("random_state", 17),
            n_jobs=model_params.get("n_jobs", -1),
        )
    else:
        raise ValueError(f"Unknown model kind: {model_params['kind']}")

    model.fit(X_train, y_train)

    # Save model & log as W&B artifact
    model_path = Path("data/06_models/model_baseline.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    artifact = wandb.Artifact("model_baseline", type="model")
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)

    return model


def evaluate(model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the model on the test set.

    Returns:
        pd.DataFrame: Evaluation metrics.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "average_precision": average_precision_score(y_test, y_proba),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    # send metrics to W&B dashboard
    wandb.log(metrics)

    metrics_df = pd.DataFrame([metrics])
    return metrics_df
