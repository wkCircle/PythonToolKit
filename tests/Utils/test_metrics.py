import numpy as np
from Utils.metrics import MetricsCls


class TestMetricsCls:
    def assert_dict_equal(self, result, expected):
        """extended version of == to check dict equality also on missing values."""
        assert (
            result.keys() == expected.keys()
        ), f"keys unmatched! \nExpected:{expected.keys} \nGet:{result.keys}"
        for k in result.keys():
            np.testing.assert_allclose(result[k], expected[k])
            # f"nan unmatched at {k}! \nExpected{expected[k]} \nGet:{result[k]}"

    @staticmethod
    def set_multioutput(evaluator: MetricsCls, multioutput: str):
        assert isinstance(multioutput, str)
        assert multioutput in ["uniform_average", "raw_values"]
        keys = [k + "__multioutput" for k in evaluator.metrics_dict.keys()]
        # INPLACELY change attribute of evaluator
        evaluator.config_dict = dict(zip(keys, [multioutput] * len(keys)))

    def test_perfect_1d(self):
        y_true = range(10)
        y_pred = range(10)
        evaluator = MetricsCls()

        # case 1: multioutput = uniform_average (default)
        expected = {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "DS": 100.0, "RMSSE": 0.0}
        self.set_multioutput(evaluator, "uniform_average")
        out = evaluator.score(y_true, y_pred)
        self.assert_dict_equal(out, expected)

        # case 2: multioutput = raw_values
        expected = {
            "MAE": np.array([0.0]),
            "RMSE": np.array([0.0]),
            "MAPE": np.array([0.0]),
            "DS": np.array([100.0]),
            "RMSSE": np.array([0.0]),
        }
        self.set_multioutput(evaluator, "raw_values")
        out = evaluator.score(y_true, y_pred)
        self.assert_dict_equal(out, expected)

    def test_perfect_2d(self):
        y_true = (np.zeros((10, 10)) + np.arange(10)).T
        y_pred = y_true.copy()
        evaluator = MetricsCls()

        # case 1: multioutput = uniform_average (default)
        expected = {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "DS": 100.0, "RMSSE": 0.0}
        self.set_multioutput(evaluator, "uniform_average")
        out = evaluator.score(y_true, y_pred)
        self.assert_dict_equal(out, expected)

        # case 2: multioutput = raw_values
        expected = {
            "MAE": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "RMSE": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "MAPE": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "DS": np.array(
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
            ),
            "RMSSE": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        self.set_multioutput(evaluator, "raw_values")
        out = evaluator.score(y_true, y_pred)
        self.assert_dict_equal(out, expected)

    def test_extreme_1d(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([0.1, 0.1])
        evaluator = MetricsCls()

        # case 1: multioutput = uniform_average (default)
        expected = {
            "MAE": 0.1,
            "RMSE": 0.1,
            "MAPE": 450359962737049.6,
            "DS": 100.0,
            "RMSSE": 6710886.4,
        }
        self.set_multioutput(evaluator, "uniform_average")
        out = evaluator.score(y_true, y_pred)
        self.assert_dict_equal(out, expected)

        # case 2: multioutput = raw_values
        expected = {
            "MAE": np.array([0.1]),
            "RMSE": np.array([0.1]),
            "MAPE": np.array([450359962737049.6]),
            "DS": np.array([100.0]),
            "RMSSE": np.array([6710886.4]),
        }
        self.set_multioutput(evaluator, "raw_values")
        out = evaluator.score(y_true, y_pred)
        self.assert_dict_equal(out, expected)

    def test_extreme_2d_singlerow(self):
        y_true = np.zeros((1, 3))
        y_pred = np.zeros((1, 3)) + 0.1

        evaluator = MetricsCls()
        # case 1: multioutput = uniform_average (default)
        expected = {
            "MAE": 0.1,
            "RMSE": 0.1,
            "MAPE": 450359962737049.7,
            "DS": np.nan,
            "RMSSE": np.nan,
        }
        self.set_multioutput(evaluator, "uniform_average")
        out = evaluator.score(y_true, y_pred)
        self.assert_dict_equal(out, expected)

        # case 2: multioutput = raw_values
        expected = {
            "MAE": np.array([0.1, 0.1, 0.1]),
            "RMSE": np.array([0.1, 0.1, 0.1]),
            "MAPE": np.array([4.50359963e14, 4.50359963e14, 4.50359963e14]),
            "DS": np.array([np.nan, np.nan, np.nan]),
            "RMSSE": np.array([np.nan, np.nan, np.nan]),
        }
        self.set_multioutput(evaluator, "raw_values")
        out = evaluator.score(y_true, y_pred)
        self.assert_dict_equal(out, expected)

    def test_extreme_2d_singlcol(self):
        y_true = np.zeros((3, 1))
        y_pred = np.zeros((3, 1)) + 0.1
        evaluator = MetricsCls()
        # case 1: multioutput = uniform_average (default)
        expected = {
            "MAE": 0.1,
            "RMSE": 0.1,
            "MAPE": 450359962737049.7,
            "DS": 100.0,
            "RMSSE": 6710886.4,
        }
        self.set_multioutput(evaluator, "uniform_average")
        out = evaluator.score(y_true, y_pred)
        self.assert_dict_equal(out, expected)
        # case 2: multioutput = raw_values
        expected = {
            "MAE": np.array([0.1]),
            "RMSE": np.array([0.1]),
            "MAPE": np.array([4.50359963e14]),
            "DS": np.array([100.0]),
            "RMSSE": np.array([6710886.4]),
        }
        self.set_multioutput(evaluator, "raw_values")
        out = evaluator.score(y_true, y_pred)
        self.assert_dict_equal(out, expected)

    def test_multioutput_1d(self):
        pass

    def test_multioutput_2d(self):
        y_true = np.zeros((5, 3)) + np.arange(1, 6).reshape(-1, 1)
        y_pred = y_true + np.arange(3)
        evaluator = MetricsCls()

        # case 1: multioutput = uniform_average (default)
        expected = {
            "MAE": 1.0,
            "RMSE": 1.0,
            "MAPE": 0.4567,
            "DS": 100.0,
            "RMSSE": 1.291,
        }
        self.set_multioutput(evaluator, "uniform_average")
        out = evaluator.score(y_true, y_pred)
        self.assert_dict_equal(out, expected)

        # case 2: multioutput = raw_values
        expected = {
            "MAE": np.array([0.0, 1.0, 2.0]),
            "RMSE": np.array([0.0, 1.0, 2.0]),
            "MAPE": np.array([0.0, 0.4567, 0.9133]),
            "DS": np.array([100.0, 100.0, 100.0]),
            "RMSSE": np.array([0.0, 1.0, 2.0]),
        }
        self.set_multioutput(evaluator, "raw_values")
        out = evaluator.score(y_true, y_pred)
