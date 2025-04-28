import unittest

from mewtwo.embeddings.terminator.a_tract import ATract
from mewtwo.embeddings.sequence import RNASequence


class TestATract(unittest.TestCase):
    def test_to_vector(self):
        a_tract_1 = ATract(RNASequence("ACGU"))

        self.assertEqual(a_tract_1.to_vector(4), [1, 0, 2,
                                                  0, 1, 3,
                                                  1, 0, 3,
                                                  0, 1, 2])
        self.assertEqual(a_tract_1.to_vector(3), [0, 1, 3,
                                                  1, 0, 3,
                                                  0, 1, 2])
        self.assertEqual(a_tract_1.to_vector(5), [0, 0, 0,
                                                  1, 0, 2,
                                                  0, 1, 3,
                                                  1, 0, 3,
                                                  0, 1, 2])

        self.assertEqual(a_tract_1.to_vector(4, one_hot=True), [1, 0, 0, 0,
                                                                0, 1, 0, 0,
                                                                0, 0, 1, 0,
                                                                0, 0, 0, 1])
        self.assertEqual(a_tract_1.to_vector(3, one_hot=True), [0, 1, 0, 0,
                                                                0, 0, 1, 0,
                                                                0, 0, 0, 1])
        self.assertEqual(a_tract_1.to_vector(5, one_hot=True), [0, 0, 0, 0,
                                                                1, 0, 0, 0,
                                                                0, 1, 0, 0,
                                                                0, 0, 1, 0,
                                                                0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
