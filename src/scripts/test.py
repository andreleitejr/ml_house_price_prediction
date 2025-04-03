import pandas as pd
from src.config import TEST_DATA_PATH, PREPROCESSOR_PATH, MODEL_PATH, PREDICTIONS_PATH
from src.scripts.data import feature_engineering
from src.scripts.model import load_model


def test():
    """Loads a trained model, preprocesses test data, makes predictions, and saves the results."""
    test_data = pd.read_csv(TEST_DATA_PATH, index_col="Id")
    test_data = feature_engineering(test_data)

    preprocessor = load_model(PREPROCESSOR_PATH)
    test_data_transformed = preprocessor.transform(test_data)

    model = load_model(MODEL_PATH)
    predictions = model.predict(test_data_transformed)

    results = pd.DataFrame({"Id": test_data.index, "SalePrice": predictions})
    results.to_csv(PREDICTIONS_PATH, index=False)


if __name__ == "__main__":
    test()
