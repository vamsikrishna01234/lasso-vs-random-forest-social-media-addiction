"""Data loading and feature preparation for the social media addiction dataset."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "social_media_addiction_dataset.csv"

TARGET = "Addiction_Level"
CLASS_NAMES = ["Low", "Moderate", "High"]  # encoded as 0, 1, 2
FREQUENCY_ORDER = {"Rarely": 0, "Occasionally": 1, "Frequently": 2, "Always": 3}

# "all"     = every behavioural column in the dataset (default)
# "report3" = only the three predictors named in the project report
FEATURE_SETS = {
    "all": [
        "Age",
        "Gender",
        "Daily_Screen_Time_Hours",
        "Num_Social_Media_Accounts",
        "Frequency_Use",
        "Posts_Per_Day",
        "Comments_Per_Day",
        "Likes_Per_Day",
        "Time_Before_Sleep_Minutes",
    ],
    "report3": ["Age", "Frequency_Use", "Num_Social_Media_Accounts"],
}


def load_dataset(path=DATA_PATH):
    """Read the CSV and run basic sanity checks."""
    df = pd.read_csv(path)
    if df.isna().any().any():
        raise ValueError("The dataset contains missing values - clean them first.")
    print(f"Loaded {len(df):,} records and {df.shape[1]} columns from {Path(path).name}")
    return df


def build_features(df, feature_set="all"):
    """Return (X, y): numeric feature table and integer class labels."""
    X = df[FEATURE_SETS[feature_set]].copy()

    if "Frequency_Use" in X.columns:
        X["Frequency_Use"] = X["Frequency_Use"].map(FREQUENCY_ORDER)
        if X["Frequency_Use"].isna().any():
            raise ValueError("Unexpected value found in Frequency_Use.")
    if "Gender" in X.columns:
        X = pd.get_dummies(X, columns=["Gender"], dtype=int)

    y = df[TARGET].map({name: i for i, name in enumerate(CLASS_NAMES)})
    if y.isna().any():
        raise ValueError("Unexpected value found in Addiction_Level.")
    return X, y.astype(int)


def split_and_scale(X, y, seed, test_size=0.30):
    """Stratified 70/30 split, then standardise using the training data only."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr.to_numpy(), y_te.to_numpy()
