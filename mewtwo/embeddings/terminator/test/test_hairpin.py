import unittest
from mewtwo.embeddings.terminator.hairpin import RNAFoldHairpin, TransTermHPHairpin


class TestHairpin(unittest.TestCase):
    def test_contains_multiple_hairpins(self):
        self.assertTrue(
            RNAFoldHairpin("example_rnafold_true", 5, -9.8, 'tggggccaaacggtgccggtaagaccacca',
                           '.((.(((....))).))(((......))).').contains_multiple_hairpins())
        self.assertTrue(
            RNAFoldHairpin("example_rnafold_true", 5, -9.8, 'ccattaggaaaggattacc',
                           '((....))...((....))').contains_multiple_hairpins())
        self.assertFalse(
            RNAFoldHairpin("example_rnafold_false", 3, -9.1, 'cccacccgaggggtggg',
                           '(((((((...)))))))').contains_multiple_hairpins())
        self.assertFalse(
            RNAFoldHairpin("example_rnafold_false", 3, -9.1, 'cccatcccgaggggtggg',
                           '((((.(((...)))))))').contains_multiple_hairpins())

        self.assertFalse(
            TransTermHPHairpin("example_transtermhp_false", 5, -9.8, 'a-tg aaa caat').contains_multiple_hairpins())
        self.assertFalse(
            TransTermHPHairpin("example_transtermhp_false", 5, -9.8, 'atg aaa cat').contains_multiple_hairpins())


if __name__ == '__main__':
    unittest.main()
