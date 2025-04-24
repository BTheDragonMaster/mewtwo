import unittest
from mewtwo.embeddings.terminator.stem import Stem
from mewtwo.embeddings.bases import BasePair, Base, PairingType
from mewtwo.embeddings.sequence import RNASequence


class TestStem(unittest.TestCase):
    def test_get_basepairs(self):
        simple_stem = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                       BasePair(Base.A, Base.U, True)]
        upstream_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                          BasePair(Base.A, Base.ZERO_PADDING, False),
                          BasePair(Base.U, Base.A, True)]
        downstream_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                            BasePair(Base.ZERO_PADDING, Base.U, False),
                            BasePair(Base.U, Base.A, True)]
        mismatch = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                    BasePair(Base.U, Base.U, False), BasePair(Base.U, Base.A, True)]
        mismatched_closing_stack = [BasePair(Base.G, Base.G, False), BasePair(Base.A, Base.U, True),
                                    BasePair(Base.A, Base.U, True), BasePair(Base.U, Base.A, True)]
        double_upstream_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                                 BasePair(Base.A, Base.ZERO_PADDING, False),
                                 BasePair(Base.A, Base.ZERO_PADDING, False), BasePair(Base.U, Base.A, True)]
        double_downstream_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                                   BasePair(Base.ZERO_PADDING, Base.U, False),
                                   BasePair(Base.ZERO_PADDING, Base.U, False), BasePair(Base.U, Base.A, True)]
        upstream_mismatch_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                                   BasePair(Base.A, Base.C, False),
                                   BasePair(Base.A, Base.ZERO_PADDING, False), BasePair(Base.U, Base.A, True)]
        downstream_mismatch_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                                     BasePair(Base.C, Base.U, False),
                                     BasePair(Base.ZERO_PADDING, Base.U, False), BasePair(Base.U, Base.A, True)]

        self.assertEqual(simple_stem, Stem(RNASequence('GAA'), '(((', RNASequence('UUC'), ')))').get_basepairs())
        self.assertEqual(upstream_bulge, Stem(RNASequence('GAAU'), '((.(', RNASequence('AUC'), ')))').get_basepairs())
        self.assertEqual(downstream_bulge, Stem(RNASequence('GAU'), '(((', RNASequence('AUUC'), ').))').get_basepairs())
        self.assertEqual(mismatch, Stem(RNASequence('GAUU'), '((.(', RNASequence('AUUC'), ').))').get_basepairs())
        self.assertEqual(mismatched_closing_stack,
                         Stem(RNASequence('GAAU'), '.(((', RNASequence('AUUG'), '))).').get_basepairs())
        self.assertEqual(double_upstream_bulge,
                         Stem(RNASequence('GAAAU'), '((..(', RNASequence('AUC'), ')))').get_basepairs())
        self.assertEqual(double_downstream_bulge,
                         Stem(RNASequence('GAU'), '(((', RNASequence('AUUUC'), ')..))').get_basepairs())
        self.assertEqual(upstream_mismatch_bulge,
                         Stem(RNASequence('GAAAU'), '((..(', RNASequence('ACUC'), ').))').get_basepairs())
        self.assertEqual(downstream_mismatch_bulge,
                         Stem(RNASequence('GACU'), '((.(', RNASequence('AUUUC'), ')..))').get_basepairs())

        with self.assertRaises(AssertionError):
            Stem(RNASequence('GAAU'), '.((', RNASequence('AUUG'), '))).')
        with self.assertRaises(AssertionError):
            Stem(RNASequence('GAAU'), '.(((', RNASequence('AUUG'), ')).')

    def test_to_vector(self):
        simple_stem = Stem(RNASequence('GAA'), '(((', RNASequence('UUC'), ')))')
        upstream_bulge = Stem(RNASequence('GAAU'), '((.(', RNASequence('AUC'), ')))')
        gu_mismatch = Stem(RNASequence('GAGU'), '((.(', RNASequence('AUUC'), ').))')

        self.assertEqual(simple_stem.to_vector(3), [2, 3, 1, 3, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1])
        self.assertEqual(simple_stem.to_vector(4), [2, 3, 1, 3, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 0, 0, 0, 0, 0])
        self.assertEqual(simple_stem.to_vector(3, one_hot=True), [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                                                                  1, 0, 0, 0, 0, 0, 0, 1, 1])
        self.assertEqual(upstream_bulge.to_vector(4, one_hot=True), [0, 0, 1, 0, 0, 1, 0, 0, 1,
                                                                     1, 0, 0, 0, 0, 0, 0, 1, 1,
                                                                     1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                     0, 0, 0, 1, 1, 0, 0, 0, 1])
        self.assertEqual(gu_mismatch.to_vector(4, pairing_type=PairingType.WATSON_CRICK),
                         [2, 3, 1, 3, 1,
                          2, 2, 1, 2, 1,
                          2, 3, 1, 2, 0,
                          1, 2, 2, 2, 1])

        self.assertEqual(gu_mismatch.to_vector(4, pairing_type=PairingType.WOBBLE_OR_WATSON_CRICK),
                         [2, 3, 1, 3, 1,
                          2, 2, 1, 2, 1,
                          2, 3, 1, 2, 1,
                          1, 2, 2, 2, 1])

        self.assertEqual(gu_mismatch.to_vector(5, pairing_type=PairingType.WOBBLE_OR_WATSON_CRICK),
                         [2, 3, 1, 3, 1,
                          2, 2, 1, 2, 1,
                          2, 3, 1, 2, 1,
                          1, 2, 2, 2, 1,
                          0, 0, 0, 0, 0])

        with self.assertRaises(AssertionError):
            upstream_bulge.to_vector(3, one_hot=True), [0, 0, 1, 0, 0, 1, 0, 0, 1,
                                                        1, 0, 0, 0, 0, 0, 0, 1, 1,
                                                        1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                        0, 0, 0, 1, 1, 0, 0, 0, 1]


if __name__ == '__main__':
    unittest.main()
