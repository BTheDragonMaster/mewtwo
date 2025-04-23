import unittest

from mewtwo.embeddings.sequence import RNASequence, DNASequence, convert_to_dna, convert_to_rna
from mewtwo.embeddings.bases import Base


class TestSequence(unittest.TestCase):
    def test_getitem(self):
        rna_sequence = RNASequence("UACCGCGAACUGG")
        dna_sequence = DNASequence("TTATCCGAGAAGC")

        self.assertEqual(RNASequence('UACC'), rna_sequence[:4])
        self.assertEqual(Base["U"], rna_sequence[0])
        self.assertEqual(RNASequence('UCG'), rna_sequence[0:5:2])
        self.assertEqual(RNASequence("GGUCAAGCGCCAU"), rna_sequence[::-1])

        self.assertEqual(DNASequence('CCGA'), dna_sequence[4:8])
        self.assertNotEqual(RNASequence('CCGA'), dna_sequence[4:8])
        self.assertEqual(Base["G"], dna_sequence[6])
        self.assertNotEqual('G', dna_sequence[6])

        with self.assertRaises(IndexError):
            _ = rna_sequence[20]

    def test_check_sequence(self):

        rna_with_t = "TACCGCGAACUGG"
        dna_with_u = "TTAUCCGAGAAGC"
        rna_with_x = "XACCGCGAACUGG"
        dna_with_x = "TTAXCCGAGAAGC"

        with self.assertRaises(ValueError):
            RNASequence(rna_with_t)
        with self.assertRaises(ValueError):
            RNASequence(dna_with_u)
        with self.assertRaises(ValueError):
            RNASequence(rna_with_x)
        with self.assertRaises(ValueError):
            DNASequence(dna_with_x)

    def test_convert_to_dna(self):
        rna_sequence = RNASequence("UACCGCGAACUGG")
        dna_sequence_correct = DNASequence("TACCGCGAACTGG")
        dna_sequence_incorrect = DNASequence("TTATCCGAGAAGC")

        self.assertEqual(dna_sequence_correct, convert_to_dna(rna_sequence))
        self.assertNotEqual(dna_sequence_incorrect, convert_to_dna(rna_sequence))

    def test_convert_to_rna(self):
        dna_sequence = DNASequence("TACCGCGAACTGG")
        rna_sequence_correct = RNASequence("UACCGCGAACUGG")
        rna_sequence_incorrect = RNASequence("UUAUCCGAGAAGC")

        self.assertEqual(rna_sequence_correct, convert_to_rna(dna_sequence))
        self.assertNotEqual(rna_sequence_incorrect, convert_to_rna(dna_sequence))


if __name__ == '__main__':
    unittest.main()
