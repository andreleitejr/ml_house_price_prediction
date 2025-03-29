import pandas as pd
from src.model import load_model, evaluate_model
from src.data_preprocessing import load_data, preprocess_data, split_data, scale_data
from sklearn.metrics import mean_absolute_error

def main():
    model = load_model('models/house_price_model.pkl')

    df = load_data('data/raw/house_prices.csv')
    df = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(df)
    x_train_scaled, x_test_scaled, _ = scale_data(x_train, x_test)

    y_pred = model.predict(x_test_scaled)

    mse = evaluate_model(model, x_test_scaled, y_test)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n### Resultados da Avaliação ###")
    print(f"- Erro Quadrático Médio (MSE): {mse}")
    print(f"- Erro Absoluto Médio (MAE): {mae}")

    results_df = pd.DataFrame({'Preço Real': y_test.values, 'Preço Previsto': y_pred})
    results_df['Diferença'] = results_df['Preço Previsto'] - results_df['Preço Real']

    print("\n### Amostra de Previsões ###")
    print(results_df.head(10).round(2))

if __name__ == '__main__':
    main()
