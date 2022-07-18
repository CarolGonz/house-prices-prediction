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
        expected_stats = self.housing_forecasting.price_stats(expected_y)
        self.assertEqual(expected_stats.__class__.__name__, "dict")

        # # Should return predictions
        model_output = self.housing_forecasting.model_predictions(expected_X, expected_y)

        self.assertCountEqual(list(model_output.keys()), ["x_train", "x_test", "y_train", "y_test", "regressor", "preds"])
        self.assertGreater(len(model_output["preds"]), 0)
        self.assertEqual(model_output["preds"].shape, model_output["y_test"].shape)

        # # Should evaluate model's performance
        r2_score = self.housing_forecasting.model_performance(model_output["y_test"], model_output["preds"])
        self.assertGreaterEqual(r2_score, 0, "R2 score should be greater than 0")
        self.assertLessEqual(r2_score, 1, "R2 score should be less than 1")


if __name__ == "__main__":
    unittest.main()
