from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class HousingForecasting:

    def load_dataset(self):
        """ Load data from the sklearn api. """
        X, y = load_boston(return_X_y=True)

        return X, y

    def model_predictions(self, X, y):
        """ Split the data in train and test samples, define a Linear regressor,
            train the regressor on the training sample and predict for the test sample.

            Return a dict with the train and test sample, the regressor and the predictions.
        """
        # split in train and test
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)

        # creat a regressor using Linear Regression
        regressor = LinearRegression()

        # Fit the model on the train sample
        regressor.fit(x_train, y_train)

        # Fit the model on the test sample
        preds = regressor.predict(x_test)

        model_output = {
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test,
            'regressor': regressor,
            'preds': preds,

        }

        return model_output

    def model_performance(self, y_true, y_predicted):
        """ Calculates and returns the r2 score between
            true (y_true) and predicted (y_predict) values.
        """

        score = round(r2_score(y_true, y_predicted), 2)

        return score


