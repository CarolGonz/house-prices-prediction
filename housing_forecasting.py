from sklearn.datasets import load_boston


class HousingForecasting:

    def load_dataset(self):
        """ Load data from the sklearn api. """
        X, y = load_boston(return_X_y=True)

        return X, y

