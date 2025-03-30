import pandas as pd
from src.model import load_model, evaluate_model
from src.data_preprocessing import load_data, preprocess_data, split_data, scale_data
from sklearn.metrics import mean_absolute_error

def main():
    model = load_model('models/house_price_model.pkl')

    home_data = load_data('data/raw/iowa_house_prices.csv')
    home_data = preprocess_data(home_data)
    train_X, val_X, train_y, val_y = split_data(home_data)
    # train_X_scaled, val_X_scaled, _ = scale_data(train_X, val_X)

    mae = evaluate_model(model, val_X, val_y)
    print(f"Mean Absolute Error in Test: {mae}")

if __name__ == '__main__':
    main()
