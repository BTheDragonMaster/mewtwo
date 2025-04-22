import unittest
from mewtwo.embeddings.terminator.stem import Stem
from mewtwo.embeddings.bases import BasePair


class TestStem(unittest.TestCase):
    def test_get_basepairs(self):
        simple_stem = [BasePair('G', 'C', True), BasePair('A', 'T', True), BasePair('A', 'T', True)]
        upstream_bulge = [BasePair('G', 'C', True), BasePair('A', 'T', True), BasePair('A', None, False),
                          BasePair('T', 'A', True)]
        downstream_bulge = [BasePair('G', 'C', True), BasePair('A', 'T', True), BasePair(None, 'T', False),
                          BasePair('T', 'A', True)]
        mismatch = [BasePair('G', 'C', True), BasePair('A', 'T', True), BasePair('T', 'T', False),
                          BasePair('T', 'A', True)]
        mismatched_closing_stack = [BasePair('G', 'G', False), BasePair('A', 'T', True), BasePair('A', 'T', True),
                                    BasePair('T', 'A', True)]
        double_upstream_bulge = [BasePair('G', 'C', True), BasePair('A', 'T', True), BasePair('A', None, False),
                                 BasePair('A', None, False), BasePair('T', 'A', True)]
        double_downstream_bulge = [BasePair('G', 'C', True), BasePair('A', 'T', True), BasePair(None, 'T', False),
                                   BasePair(None, 'T', False), BasePair('T', 'A', True)]
        upstream_mismatch_bulge = [BasePair('G', 'C', True), BasePair('A', 'T', True), BasePair('A', 'C', False),
                                   BasePair('A', None, False), BasePair('T', 'A', True)]
        downstream_mismatch_bulge = [BasePair('G', 'C', True), BasePair('A', 'T', True), BasePair('C', 'T', False),
                                     BasePair(None, 'T', False), BasePair('T', 'A', True)]

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
