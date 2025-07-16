import unittest
import os
from shutil import rmtree


from mewtwo.machine_learning.data_preparation.train_test_split import split_data_from_file
from mewtwo.parsers.parse_dnabert_data import parse_dnabert_data

BASE_DIR = os.path.dirname(__file__)


class TestTrainTestSplit(unittest.TestCase):

    def test_split_data_from_file(self):


        test_data_1 = os.path.abspath(os.path.join(BASE_DIR, 'data', 'mock_data_1.txt'))
        test_output_1 = os.path.abspath(os.path.join(BASE_DIR, 'output', 'mock_output_1'))

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
