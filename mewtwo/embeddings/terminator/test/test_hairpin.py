import unittest
from mewtwo.embeddings.terminator.hairpin import RNAFoldHairpin, TransTermHPHairpin


class TestHairpin(unittest.TestCase):
    def test_contains_multiple_hairpins(self):
        self.assertTrue(
            RNAFoldHairpin("example_rnafold_true", -9.8, 'tggggccaaacggtgccggtaagaccacca',
                           '.((.(((....))).))(((......))).', 5).contains_multiple_hairpins())
        self.assertTrue(
            RNAFoldHairpin("example_rnafold_true", -9.8, 'ccattaggaaaggattacc',
                           '((....))...((....))', 5).contains_multiple_hairpins())
        self.assertFalse(
            RNAFoldHairpin("example_rnafold_false", -9.1, 'cccacccgaggggtggg',
                           '(((((((...)))))))', 3).contains_multiple_hairpins())
        self.assertFalse(
            RNAFoldHairpin("example_rnafold_false", -9.1, 'cccatcccgaggggtggg',
                           '((((.(((...)))))))', 3).contains_multiple_hairpins())

        self.assertFalse(
            TransTermHPHairpin("example_transtermhp_false", -9.8, 'a-tg aaa caat', 5).contains_multiple_hairpins())
        self.assertFalse(
            TransTermHPHairpin("example_transtermhp_false", -9.8, 'atg aaa cat', 5).contains_multiple_hairpins())


if __name__ == '__main__':
    unittest.main()
