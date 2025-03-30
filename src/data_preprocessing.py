import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(file_path)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns, handle missing numerical columns by filling with the median, and convert categorical features to numeric."""
    # data = data.dropna(subset=['price'])
    # data = data.fillna(data.median())
    #
    # data = pd.get_dummies(data, drop_first=True)

    return data

def get_y(data: pd.DataFrame):
    return data.SalePrice

def get_X(data: pd.DataFrame):
    features =  ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    return data[features]

def get_X_y(data: pd.DataFrame):
    return get_X(data), get_y(data)

def split_data(data: pd.DataFrame):
    """Split the data into training and testing datasets."""
    y = get_y(data)
    X = get_X(data)
    return train_test_split(X, y, random_state=0)


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
