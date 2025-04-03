from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_DATA_PATH = DATA_DIR / "train" / "train_house_prices.csv"
TEST_DATA_PATH = DATA_DIR / "test" / "test_house_prices.csv"
MODEL_PATH = MODEL_DIR / "trained_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
