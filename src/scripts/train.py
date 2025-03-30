from src.scripts.data import load_data, preprocess_data, split_data
from src.scripts.model import train_model, evaluate_model, save_model

def main():
    home_data = load_data('src/data/raw/iowa_house_prices.csv')
    home_data = preprocess_data(home_data)
    train_X, val_X, train_y, val_y = split_data(home_data)

    model = train_model(train_X, train_y)
    mae = evaluate_model(model, val_X, val_y)
    print(f'Mean Absolute Error: {mae}')

    save_model(model, 'src/models/house_prices_model.pkl')

if __name__ == '__main__':
    main()
