from src.data_preprocessing import load_data, preprocess_data, split_data, scale_data
from src.model import train_model, evaluate_model, save_model

def main():
    home_data = load_data('data/raw/iowa_house_prices.csv')

    home_data = preprocess_data(home_data)

    train_X, val_X, train_y, val_y = split_data(home_data)

    # x_train_scaled, x_test_scaled, scaler = scale_data(x_train, x_test)

    model = train_model(train_X, train_y)
    mae = evaluate_model(model, val_X, val_y)
    print(f'Mean Absolute Error: {mae}')

    save_model(model, 'models/house_price_model.pkl')

if __name__ == '__main__':
    main()
