import pandas as pd
from src.scripts.data import load_data, split_data, preprocess_data
from src.scripts.model import train_model, evaluate_model, save_model


def main():
    data = load_data("src/datasets/train/train_house_prices.csv")
    X_train, X_valid, y_train, y_valid = split_data(data)
    X_train, X_valid = preprocess_data(X_train, X_valid)

    model = train_model(X_train, X_valid, y_train, y_valid)
    mae = evaluate_model(model, X_valid, y_valid)

    print(f"Mean Absolute Error: {mae:.2f}")

    save_model(model, "src/models/trained_model.pkl")


if __name__ == "__main__":
    main()
