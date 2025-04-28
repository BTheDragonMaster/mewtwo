import unittest

from mewtwo.embeddings.terminator.u_tract import UTract
from mewtwo.embeddings.sequence import RNASequence


class TestUTract(unittest.TestCase):
    def test_to_vector(self):
        u_tract_1 = UTract(RNASequence("ACGU"), 2)

        self.assertEqual(u_tract_1.to_vector(4), [1, 0, 2, 0,
                                                  0, 1, 3, 0,
                                                  1, 0, 3, 1,
                                                  0, 1, 2, 0])
        self.assertEqual(u_tract_1.to_vector(3), [1, 0, 2, 0,
                                                  0, 1, 3, 0,
                                                  1, 0, 3, 1])
        self.assertEqual(u_tract_1.to_vector(5), [1, 0, 2, 0,
                                                  0, 1, 3, 0,
                                                  1, 0, 3, 1,
                                                  0, 1, 2, 0,
                                                  0, 0, 0, 0])

        self.assertEqual(u_tract_1.to_vector(4, one_hot=True), [1, 0, 0, 0, 0,
                                                                0, 1, 0, 0, 0,
                                                                0, 0, 1, 0, 1,
                                                                0, 0, 0, 1, 0])
        self.assertEqual(u_tract_1.to_vector(3, one_hot=True), [1, 0, 0, 0, 0,
                                                                0, 1, 0, 0, 0,
                                                                0, 0, 1, 0, 1])
        self.assertEqual(u_tract_1.to_vector(5, one_hot=True), [1, 0, 0, 0, 0,
                                                                0, 1, 0, 0, 0,
                                                                0, 0, 1, 0, 1,
                                                                0, 0, 0, 1, 0,
                                                                0, 0, 0, 0, 0])

    def test_init(self):
        u_tract_1 = UTract(RNASequence("ACGU"), 2)
        self.assertEqual(2, u_tract_1.pot)

        with self.assertRaises(AssertionError):
            UTract(RNASequence("ACGU"), 4)


if __name__ == "__main__":
    unittest.main()
