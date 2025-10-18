import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def load_raw() -> pd.DataFrame:
    """Load raw data from as CSV file.

    Returns:
        pd.DataFrame: Raw data.
    """
    return pd.read_csv("data/01_raw/data.csv", skipinitialspace=True)


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
    df: pd.DataFrame,
) -> list[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the cleaned data into training and testing sets.

    Returns:
        list[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    y = df["income_encoded"]
    X = df.drop(columns=["income_encoded"])

    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, test_size=0.3, random_state=17, stratify=y
    )
    return [X_train, X_test, y_train, y_test]


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series):
    """Train a baseline model that predicts the majority class.

    Returns:
        float: Baseline accuracy.
    """
    rf_model = RandomForestClassifier(
        n_estimators=200,  # number of trees in the forest
        max_depth=None,
        random_state=17,  # to keep results consistent
        n_jobs=-1,  # use all available cores
    )

    fitted_model = rf_model.fit(X_train, y_train)

    return fitted_model


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate the model on the test set.

    Returns:
        float: Accuracy of the model on the test set.
    """
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of the positive class
    y_pred = model.predict(X_test)

    avg_precision = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    return {
        "average_precision": avg_precision,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }
