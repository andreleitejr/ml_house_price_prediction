import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""

    return pd.read_csv(file_path)


def create_result_data(data: pd.DataFrame, predictions: pd.Series) -> None:
    """Generate and save a CSV file containing the prediction results."""

    output = pd.DataFrame({'Id': data.Id, 'SalePrice': predictions})
    output.to_csv('src/datasets/result/predictions_house_prices.csv', index=False)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing numerical columns by filling with the median."""

    data.drop(['SalePrice'], axis=1)

    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    return data


def reduce_data(X_train, X_valid):
    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
    return X_train.drop(cols_with_missing, axis=1), X_valid.drop(cols_with_missing, axis=1)


def impute_data(X_train, X_valid):
    imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    imputed_X_valid = pd.DataFrame(imputer.transform(X_valid), columns=X_valid.columns)
    return imputed_X_train, imputed_X_valid


def impute_extension_data(X_train, X_valid):
    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()

    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

    return impute_data(X_train_plus, X_valid_plus)


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

    X = data.select_dtypes(exclude=['object'])
    y = data.SalePrice

    return train_test_split(X, y, random_state=0)
