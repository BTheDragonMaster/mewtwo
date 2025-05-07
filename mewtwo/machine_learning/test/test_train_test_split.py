import unittest
import os
from shutil import rmtree


from mewtwo.machine_learning.train_test_split import split_data, split_data_from_file, bin_data
from mewtwo.parsers.parse_dnabert_data import parse_dnabert_data

BASE_DIR = os.path.dirname(__file__)


class TestTrainTestSplit(unittest.TestCase):
    def test_bin_data(self):
        dummy_data_1 = [0.0, 0.05, 0.1, 0.11, 0.15, 0.9, 0.95, 1.0]
        dummy_bins_1 = [0, 0, 0, 1, 1, 8, 9, 9]

        self.assertEqual(bin_data(dummy_data_1), dummy_bins_1)

        dummy_data_2 = [1.0, 0.0, 0.05, 0.1, 0.11, 0.15, 0.9, 0.95, 1.0]
        dummy_bins_2 = [9, 0, 0, 0, 1, 1, 8, 9, 9]

        self.assertEqual(bin_data(dummy_data_2), dummy_bins_2)

        dummy_data_3 = [1.1, 0.0, 0.05, 0.1, 0.11, 0.15, 0.9, 0.95, 1.0]

        with self.assertRaises(ValueError):
            bin_data(dummy_data_3)

        dummy_data_4 = [-0.1, 0.0, 0.05, 0.1, 0.11, 0.15, 0.9, 0.95, 1.0]

        with self.assertRaises(ValueError):
            bin_data(dummy_data_4)

    def test_split_data_from_file(self):


        test_data_1 = os.path.abspath(os.path.join(BASE_DIR, 'data', 'mock_data_1.txt'))
        test_output_1 = os.path.abspath(os.path.join(BASE_DIR, 'output', 'mock_output_1'))

        test_data_2 = os.path.abspath(os.path.join(BASE_DIR, 'data', 'mock_data_2.txt'))
        test_output_2 = os.path.abspath(os.path.join(BASE_DIR, 'output', 'mock_output_2'))

        split_data_from_file(test_data_1, test_output_1)

        self.assertEqual(os.path.exists(test_output_1), True)
        self.assertEqual(os.path.exists(os.path.join(test_output_1, "validation.txt")), True)
        self.assertEqual(os.path.exists(os.path.join(test_output_1, "train.txt")), True)
        self.assertEqual(os.path.exists(os.path.join(test_output_1, "test.txt")), True)

        x_train_1, y_train_1 = parse_dnabert_data(os.path.join(test_output_1, "train.txt"))
        self.assertEqual(len(x_train_1), 50)

        x_test_1, y_test_1 = parse_dnabert_data(os.path.join(test_output_1, "test.txt"))
        self.assertEqual(len(x_test_1), 25)

        x_val_1, y_val_1 = parse_dnabert_data(os.path.join(test_output_1, "validation.txt"))
        self.assertEqual(len(x_val_1), 25)

        rmtree(test_output_1)
