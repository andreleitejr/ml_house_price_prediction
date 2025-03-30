import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(file_path)


def preprocess_data(home_data: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns, handle missing numerical columns by filling with the median, and convert categorical features to numeric."""
    # home_data = home_data.dropna(subset=['price'])
    # home_data = home_data.fillna(home_data.median())
    #
    # home_data = pd.get_dummies(home_data, drop_first=True)

    return home_data


def split_data(home_data: pd.DataFrame, test_size: float = 0.4):
    """Split the data into training and testing datasets."""
    y = home_data.SalePrice
    feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[feature_columns]

    return train_test_split(X, y, random_state=0)


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
