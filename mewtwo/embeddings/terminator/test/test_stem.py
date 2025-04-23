import unittest
from mewtwo.embeddings.terminator.stem import Stem
from mewtwo.embeddings.bases import BasePair, Base


class TestStem(unittest.TestCase):
    def test_get_basepairs(self):
        simple_stem = [BasePair(Base['G'], Base['C'], True), BasePair(Base['A'], Base['T'], True),
                       BasePair(Base['A'], Base['T'], True)]
        upstream_bulge = [BasePair(Base['G'], Base['C'], True), BasePair(Base['A'], Base['T'], True),
                          BasePair(Base['A'], None, False),
                          BasePair(Base['T'], Base['A'], True)]
        downstream_bulge = [BasePair(Base['G'], Base['C'], True), BasePair(Base['A'], Base['T'], True),
                            BasePair(None, Base['T'], False),
                            BasePair(Base['T'], Base['A'], True)]
        mismatch = [BasePair(Base['G'], Base['C'], True), BasePair(Base['A'], Base['T'], True),
                    BasePair(Base['T'], Base['T'], False), BasePair(Base['T'], Base['A'], True)]
        mismatched_closing_stack = [BasePair(Base['G'], Base['G'], False), BasePair(Base['A'], Base['T'], True),
                                    BasePair(Base['A'], Base['T'], True), BasePair(Base['T'], Base['A'], True)]
        double_upstream_bulge = [BasePair(Base['G'], Base['C'], True), BasePair(Base['A'], Base['T'], True),
                                 BasePair(Base['A'], None, False),
                                 BasePair(Base['A'], None, False), BasePair(Base['T'], Base['A'], True)]
        double_downstream_bulge = [BasePair(Base['G'], Base['C'], True), BasePair(Base['A'], Base['T'], True),
                                   BasePair(None, Base['T'], False),
                                   BasePair(None, Base['T'], False), BasePair(Base['T'], Base['A'], True)]
        upstream_mismatch_bulge = [BasePair(Base['G'], Base['C'], True), BasePair(Base['A'], Base['T'], True),
                                   BasePair(Base['A'], Base['C'], False),
                                   BasePair(Base['A'], None, False), BasePair(Base['T'], Base['A'], True)]
        downstream_mismatch_bulge = [BasePair(Base['G'], Base['C'], True), BasePair(Base['A'], Base['T'], True),
                                     BasePair(Base['C'], Base['T'], False),
                                     BasePair(None, Base['T'], False), BasePair(Base['T'], Base['A'], True)]

        self.assertEqual(simple_stem, Stem('GAA', '(((', 'TTC', ')))').get_basepairs())
        self.assertEqual(upstream_bulge, Stem('GAAT', '((.(', 'ATC', ')))').get_basepairs())
        self.assertEqual(downstream_bulge, Stem('GAT', '(((', 'ATTC', ').))').get_basepairs())
        self.assertEqual(mismatch, Stem('GATT', '((.(', 'ATTC', ').))').get_basepairs())
        self.assertEqual(mismatched_closing_stack, Stem('GAAT', '.(((', 'ATTG', '))).').get_basepairs())
        self.assertEqual(double_upstream_bulge, Stem('GAAAT', '((..(', 'ATC', ')))').get_basepairs())
        self.assertEqual(double_downstream_bulge, Stem('GAT', '(((', 'ATTTC', ')..))').get_basepairs())
        self.assertEqual(upstream_mismatch_bulge, Stem('GAAAT', '((..(', 'ACTC', ').))').get_basepairs())
        self.assertEqual(downstream_mismatch_bulge, Stem('GACT', '((.(', 'ATTTC', ')..))').get_basepairs())

        with self.assertRaises(AssertionError):
            Stem('GAAT', '.((', 'ATTG', '))).')
        with self.assertRaises(AssertionError):
            Stem('GAAT', '.(((', 'ATTG', ')).')

    def test_to_vector(self):
        pass


if __name__ == '__main__':
    unittest.main()
