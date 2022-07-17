import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()