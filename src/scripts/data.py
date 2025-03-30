import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""

    return pd.read_csv(file_path)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing numerical columns by filling with the median."""

    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    return data


def get_target(data: pd.DataFrame) -> pd.Series:
    """Extract the target variable (SalePrice) from the dataset."""

    return data.SalePrice


def get_features(data: pd.DataFrame) -> pd.DataFrame:
    """Select the relevant features for model training."""

    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

    return data[features]


def get_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Retrieve both the feature set and target variable from the dataset."""

    return get_features(data), get_target(data)


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets."""

    X, y = get_features(data), get_target(data)

    return train_test_split(X, y, random_state=0)


def create_result_data(data: pd.DataFrame, predictions: pd.Series) -> None:
    """Generate and save a CSV file containing the prediction results."""

    output = pd.DataFrame({'Id': data.Id, 'SalePrice': predictions})
    output.to_csv('src/data/result/predictions_house_prices.csv', index=False)
