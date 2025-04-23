import unittest
from mewtwo.embeddings.terminator.stem import Stem
from mewtwo.embeddings.bases import BasePair, Base
from mewtwo.embeddings.sequence import RNASequence


class TestStem(unittest.TestCase):
    def test_get_basepairs(self):
        simple_stem = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                       BasePair(Base.A, Base.U, True)]
        upstream_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                          BasePair(Base.A, None, False),
                          BasePair(Base.U, Base.A, True)]
        downstream_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                            BasePair(None, Base.U, False),
                            BasePair(Base.U, Base.A, True)]
        mismatch = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                    BasePair(Base.U, Base.U, False), BasePair(Base.U, Base.A, True)]
        mismatched_closing_stack = [BasePair(Base.G, Base.G, False), BasePair(Base.A, Base.U, True),
                                    BasePair(Base.A, Base.U, True), BasePair(Base.U, Base.A, True)]
        double_upstream_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                                 BasePair(Base.A, None, False),
                                 BasePair(Base.A, None, False), BasePair(Base.U, Base.A, True)]
        double_downstream_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                                   BasePair(None, Base.U, False),
                                   BasePair(None, Base.U, False), BasePair(Base.U, Base.A, True)]
        upstream_mismatch_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                                   BasePair(Base.A, Base.C, False),
                                   BasePair(Base.A, None, False), BasePair(Base.U, Base.A, True)]
        downstream_mismatch_bulge = [BasePair(Base.G, Base.C, True), BasePair(Base.A, Base.U, True),
                                     BasePair(Base.C, Base.U, False),
                                     BasePair(None, Base.U, False), BasePair(Base.U, Base.A, True)]

        self.assertEqual(simple_stem, Stem(RNASequence('GAA'), '(((', RNASequence('UUC'), ')))').get_basepairs())
        self.assertEqual(upstream_bulge, Stem(RNASequence('GAAU'), '((.(', RNASequence('AUC'), ')))').get_basepairs())
        self.assertEqual(downstream_bulge, Stem(RNASequence('GAU'), '(((', RNASequence('AUUC'), ').))').get_basepairs())
        self.assertEqual(mismatch, Stem(RNASequence('GAUU'), '((.(', RNASequence('AUUC'), ').))').get_basepairs())
        self.assertEqual(mismatched_closing_stack, Stem(RNASequence('GAAU'), '.(((', RNASequence('AUUG'), '))).').get_basepairs())
        self.assertEqual(double_upstream_bulge, Stem(RNASequence('GAAAU'), '((..(', RNASequence('AUC'), ')))').get_basepairs())
        self.assertEqual(double_downstream_bulge, Stem(RNASequence('GAU'), '(((', RNASequence('AUUUC'), ')..))').get_basepairs())
        self.assertEqual(upstream_mismatch_bulge, Stem(RNASequence('GAAAU'), '((..(', RNASequence('ACUC'), ').))').get_basepairs())
        self.assertEqual(downstream_mismatch_bulge, Stem(RNASequence('GACU'), '((.(', RNASequence('AUUUC'), ')..))').get_basepairs())

        with self.assertRaises(AssertionError):
            Stem(RNASequence('GAAU'), '.((', RNASequence('AUUG'), '))).')
        with self.assertRaises(AssertionError):
            Stem(RNASequence('GAAU'), '.(((', RNASequence('AUUG'), ')).')

    def test_to_vector(self):
        pass


if __name__ == '__main__':
    unittest.main()
