import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path, index_col="Id")


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and validation sets after dropping null target values."""
    data = data.dropna(subset=["SalePrice"])
    y = data.pop("SalePrice")
    return train_test_split(data, y, train_size=0.8, test_size=0.2, random_state=0)


def preprocess_data(X_train: pd.DataFrame, X_valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess categorical and numerical features and align train and validation sets."""
    categorical_cols = X_train.select_dtypes(include=["object"]).nunique()[lambda x: x < 10].index.tolist()
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    selected_cols = categorical_cols + numeric_cols

    X_train = pd.get_dummies(X_train[selected_cols])
    X_valid = pd.get_dummies(X_valid[selected_cols])

    X_train, X_valid = X_train.align(X_valid, join="left", axis=1, fill_value=0)

    return X_train, X_valid


def export_data(data: dict, filepath: str) -> None:
    """Save a dictionary as a CSV file."""
    pd.DataFrame(data).to_csv(filepath, index=False)
