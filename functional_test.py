import unittest

from housing_forecasting import HousingForecasting


class FunctionalTestCase(unittest.TestCase):

    def setUp(self):
        self.housing_forecasting = HousingForecasting()

    def test_main(self):
        # # Should load the dataset from the sklearn api
        expected_X, expected_y = self.housing_forecasting.load_dataset()
        self.assertEqual(expected_X.shape[1], 13, "Expected 13 features")
        self.assertEqual(len(expected_y.shape), 1, "Expected array of shape 1, containing only the target")
        self.assertGreater(expected_X.shape[0], 0, "Expected dataset with more than 0 rows for the feature dataset")
        self.assertGreater(expected_y.shape[0], 0, "Expected dataset with more than 0 rows for the target dataset")

        # # Should do some analysis

        # # Should return predictions

        # # Should evaluate model's performance


if __name__ == "__main__":
    unittest.main()
