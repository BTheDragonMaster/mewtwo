import unittest

from mewtwo.embeddings.sequence import RNASequence
from mewtwo.embeddings.terminator.loop import Loop


class TestLoop(unittest.TestCase):
    def test_to_vector(self):
        loop_01 = Loop(RNASequence("AGCGU"), '.....')

        self.assertEqual(loop_01.to_vector(max_loop_size=5), [1, 0, 2,
                                                              1, 0, 3,
                                                              0, 1, 3,
                                                              1, 0, 3,
                                                              0, 1, 2])

        self.assertEqual(loop_01.to_vector(max_loop_size=9), [0, 0, 0,
                                                              0, 0, 0,
                                                              1, 0, 2,
                                                              1, 0, 3,
                                                              0, 1, 3,
                                                              1, 0, 3,
                                                              0, 1, 2,
                                                              0, 0, 0,
                                                              0, 0, 0])

        loop_02 = Loop(RNASequence("AGCCGU"), '......')

        with self.assertRaises(AssertionError):
            loop_02.to_vector(max_loop_size=5)

        with self.assertRaises(AssertionError):
            loop_02.to_vector(max_loop_size=6)

        self.assertEqual(loop_02.to_vector(max_loop_size=7), [1, 0, 2,
                                                              1, 0, 3,
                                                              0, 1, 3,
                                                              0, 0, 0,
                                                              0, 1, 3,
                                                              1, 0, 3,
                                                              0, 1, 2])

        self.assertEqual(loop_02.to_vector(max_loop_size=9), [0, 0, 0,
                                                              1, 0, 2,
                                                              1, 0, 3,
                                                              0, 1, 3,
                                                              0, 0, 0,
                                                              0, 1, 3,
                                                              1, 0, 3,
                                                              0, 1, 2,
                                                              0, 0, 0])


if __name__ == '__main__':
    unittest.main()
