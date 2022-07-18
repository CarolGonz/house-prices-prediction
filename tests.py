import unittest
from unittest.mock import patch

import numpy as np

from housing_forecasting import HousingForecasting


class HousingForecastingTesteCase(unittest.TestCase):

    def setUp(self):
        self.housing_forecasting = HousingForecasting()

    @patch("housing_forecasting.load_boston")
    def test_load_dataset(self, mock_load_boston):
        mock_load_boston.return_value = "X_features", "y_target"
        expected_X, expected_y = self.housing_forecasting.load_dataset()

        self.assertEqual(expected_X, "X_features")
        self.assertEqual(expected_y, "y_target")

    def test_price_stats(self):
        prices = np.array([1.0, 2.0, 6.0])

        stats_dict = self.housing_forecasting.price_stats(prices)
        self.assertCountEqual(list(stats_dict), ["min_price", "max_price", "mean_price", "median_price", "std_price"])
        self.assertEqual(stats_dict["min_price"], 1.0)
        self.assertEqual(stats_dict["max_price"], 6.0)
        self.assertEqual(stats_dict["mean_price"], 3.0)
        self.assertEqual(stats_dict["median_price"], 2.0)
        self.assertEqual(stats_dict["std_price"], 2.2)


if __name__ == "__main__":
    unittest.main()