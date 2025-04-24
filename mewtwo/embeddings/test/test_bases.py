import unittest

from mewtwo.embeddings.bases import Base, base_to_vector, BasePair, PairingType


class TestBase(unittest.TestCase):
    def test_to_vector(self):
        self.assertEqual(base_to_vector(Base.A), [2, 2])
        self.assertEqual(base_to_vector(Base.U), [1, 2])
        self.assertEqual(base_to_vector(Base.G), [2, 3])
        self.assertEqual(base_to_vector(Base.C), [1, 3])
        self.assertEqual(base_to_vector(Base.T), [1, 2])
        self.assertEqual(base_to_vector(Base.ZERO_PADDING), [0, 0])

        self.assertEqual(base_to_vector(Base.A, one_hot=True), [1, 0, 0, 0])
        self.assertEqual(base_to_vector(Base.U, one_hot=True), [0, 0, 0, 1])
        self.assertEqual(base_to_vector(Base.G, one_hot=True), [0, 0, 1, 0])
        self.assertEqual(base_to_vector(Base.C, one_hot=True), [0, 1, 0, 0])
        self.assertEqual(base_to_vector(Base.T, one_hot=True), [0, 0, 0, 1])
        self.assertEqual(base_to_vector(Base.ZERO_PADDING, one_hot=True), [0, 0, 0, 0])

        self.assertNotEqual(base_to_vector(Base.A, one_hot=True), [2, 2])

        with self.assertRaises(ValueError):
            base_to_vector(Base.DNA)


class TestBasePair(unittest.TestCase):
    def test_is_watson_crick(self):
        self.assertTrue(BasePair(Base.C, Base.G, True).is_watson_crick())
        self.assertTrue(BasePair(Base.G, Base.C, True).is_watson_crick())
        self.assertTrue(BasePair(Base.A, Base.T, True).is_watson_crick())
        self.assertTrue(BasePair(Base.T, Base.A, True).is_watson_crick())
        self.assertTrue(BasePair(Base.A, Base.U, True).is_watson_crick())
        self.assertTrue(BasePair(Base.U, Base.A, True).is_watson_crick())
        self.assertFalse(BasePair(Base.G, Base.U, True).is_watson_crick())
        self.assertFalse(BasePair(Base.U, Base.U, True).is_watson_crick())

    def test_is_wobble(self):
        self.assertTrue(BasePair(Base.G, Base.U, True).is_wobble())
        self.assertTrue(BasePair(Base.U, Base.G, True).is_wobble())
        self.assertFalse(BasePair(Base.G, Base.T, True).is_wobble())
        self.assertFalse(BasePair(Base.T, Base.G, True).is_wobble())

    def test_to_vector(self):
        base_pair_1 = BasePair(Base.G, Base.U, False)
        base_pair_2 = BasePair(Base.A, Base.U, True)

        self.assertEqual(base_pair_1.to_vector(), [2, 3, 1, 2, 0])
        self.assertEqual(base_pair_1.to_vector(one_hot=True), [0, 0, 1, 0, 0, 0, 0, 1, 0])
        self.assertEqual(base_pair_1.to_vector(pairing_type=PairingType.WOBBLE_OR_WATSON_CRICK), [2, 3, 1, 2, 1])
        self.assertEqual(base_pair_1.to_vector(pairing_type=PairingType.WATSON_CRICK), [2, 3, 1, 2, 0])

        with self.assertRaises(ValueError):
            base_pair_1.to_vector(pairing_type=PairingType.WOBBLE)

        self.assertEqual(base_pair_2.to_vector(), [2, 2, 1, 2, 1])
        self.assertEqual(base_pair_2.to_vector(one_hot=True), [1, 0, 0, 0, 0, 0, 0, 1, 1])
        self.assertEqual(base_pair_2.to_vector(pairing_type=PairingType.WOBBLE_OR_WATSON_CRICK), [2, 2, 1, 2, 1])
        self.assertEqual(base_pair_2.to_vector(pairing_type=PairingType.WATSON_CRICK), [2, 2, 1, 2, 1])


if __name__ == '__main__':
    unittest.main()
