from housing_forecasting import HousingForecasting


def main():
    housing_forecasting = HousingForecasting()

    X, y = housing_forecasting.load_dataset()

    stats = housing_forecasting.price_stats(y)
    print(f"Price stats: {stats}")

    model_output = housing_forecasting.model_predictions(X, y)

    r2_score = housing_forecasting.model_performance(model_output["y_test"], model_output["preds"])
    print(f"R2 Score: {r2_score}")


if __name__ == "__main__":
    main()