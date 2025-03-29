import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns, handle missing numerical columns by filling with the median, and convert categorical features to numeric."""
    df = df.dropna(subset=['price'])
    df = df.fillna(df.median())
    df = pd.get_dummies(df, drop_first=True)

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Split the data into training and testing datasets."""
    x = df.drop(columns='price')
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    return x_train, x_test, y_train, y_test


def scale_data(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple:
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, scaler
