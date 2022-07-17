import numpy as np
from sklearn.datasets import load_boston


class HousingForecasting:

    def load_dataset(self):
        """ Load data from the sklearn api. """
        X, y = load_boston(return_X_y=True)

        return X, y

    @staticmethod
    def price_stats(prices):
        """ Calculate some basic statistics of house prices
            Returns a dictionary with the statistics values. """

        # Minimum price of the data
        min_price = np.min(prices)

        # Maximum price of the data
        max_price = np.max(prices)

        # Mean price of the data
        mean_price = np.mean(prices)

        # Median price of the data
        median_price = np.median(prices)

        # Standard deviation of prices of the data
        std_price = np.round(np.std(prices), 1)

        price_stats = {
            "min_price": min_price,
            "max_price": max_price,
            "mean_price": mean_price,
            "median_price": median_price,
            "std_price": std_price
        }

        return price_stats
