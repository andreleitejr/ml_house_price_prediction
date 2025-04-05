from src.utils.logger import logger
from src.utils.config import TRAIN_DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH
from src.scripts.data import load_data, split_data, preprocess_data, feature_engineering
from src.scripts.model import train_model, evaluate_model, save_model, cross_validate_model

def train():
    """Executes the full pipeline: loads data, applies feature engineering,
    splits into train/validation, preprocesses features, trains the model,
    evaluates performance, performs cross-validation, and saves the trained model."""
    data = load_data(TRAIN_DATA_PATH)
    data = feature_engineering(data)

    X_train, X_valid, y_train, y_valid = split_data(data)
    X_train, X_valid = preprocess_data(X_train, X_valid, PREPROCESSOR_PATH)

    model = train_model(X_train, X_valid, y_train, y_valid)
    cross_validate_model(model, X_train, y_train)

    save_model(model, MODEL_PATH)

    logger.info('Training pipeline completed.')

if __name__ == '__main__':
    train()
