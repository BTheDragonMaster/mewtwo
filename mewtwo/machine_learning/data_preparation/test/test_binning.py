import unittest

from mewtwo.machine_learning.data_preparation.binning import bin_data

class TestBinning(unittest.TestCase):
    def test_bin_data(self):
        dummy_data_1 = [0.0, 0.05, 0.1, 0.11, 0.15, 0.9, 0.95, 1.0]
        dummy_bins_1 = [0, 0, 0, 1, 1, 8, 9, 9]

        self.assertEqual(bin_data(dummy_data_1, n_bins=10), dummy_bins_1)

        dummy_data_2 = [1.0, 0.0, 0.05, 0.1, 0.11, 0.15, 0.9, 0.95, 1.0]
        dummy_bins_2 = [9, 0, 0, 0, 1, 1, 8, 9, 9]

        self.assertEqual(bin_data(dummy_data_2, n_bins=10), dummy_bins_2)

        dummy_data_3 = [1.1, 0.0, 0.05, 0.1, 0.11, 0.15, 0.9, 0.95, 1.0]

        with self.assertRaises(ValueError):
            bin_data(dummy_data_3)

        dummy_data_4 = [-0.1, 0.0, 0.05, 0.1, 0.11, 0.15, 0.9, 0.95, 1.0]

        with self.assertRaises(ValueError):
            bin_data(dummy_data_4)

if __name__ == "__main__":
    unittest.main()