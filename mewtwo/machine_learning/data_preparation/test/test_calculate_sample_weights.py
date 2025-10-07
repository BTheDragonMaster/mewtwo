import unittest
from math import isclose

from mewtwo.machine_learning.data_preparation.calculate_sample_weights import get_sample_weights


class CalculateSampleWeights(unittest.TestCase):
    def test_get_sample_weights(self):
        dataset_1 = [0.1, 0.3, 0.5, 0.7, 0.9]
        weights_1 = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.assertNearlyEqual(weights_1, get_sample_weights(dataset_1))

        dataset_2 = [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.5, 0.7, 0.9, 0.9]
        weights_2 = [1.0, 1.0, 1.0, 1.0, 0.66666, 0.666666, 0.66666, 2.0, 1.0, 1.0]

        self.assertNearlyEqual(weights_2, get_sample_weights(dataset_2))

    def assertNearlyEqual(self, list_1, list_2):
        for i, element_1 in enumerate(list_1):
            element_2 = list_2[i]
            if not isclose(element_1, element_2, abs_tol=0.00001):
                self.fail(f"Lists are not equal: {list_1}, {list_2}. \n First mismatching element: {i} ([{element_1}], [{element_2}])")


if __name__ == "__main__":
    unittest.main()