import unittest
from mewtwo.embeddings.feature_labels import FeatureLabel
from mewtwo.embeddings.bases import Base


class TestFeatureLabel(unittest.TestCase):

    def test_set_base_information(self):

        # A-tract features

        feature_a01 = FeatureLabel(0, 10, 1, 1, 1)
        feature_a02 = FeatureLabel(1, 10, 1, 1, 1)
        feature_a03 = FeatureLabel(2, 10, 1, 1, 1)
        feature_a04 = FeatureLabel(3, 10, 1, 1, 1)
        feature_a05 = FeatureLabel(4, 10, 1, 1, 1)
        feature_a06 = FeatureLabel(5, 10, 1, 1, 1)

        feature_a07 = FeatureLabel(0, 10, 1, 1, 1, one_hot=True)
        feature_a08 = FeatureLabel(1, 10, 1, 1, 1, one_hot=True)
        feature_a09 = FeatureLabel(2, 10, 1, 1, 1, one_hot=True)
        feature_a10 = FeatureLabel(3, 10, 1, 1, 1, one_hot=True)
        feature_a11 = FeatureLabel(4, 10, 1, 1, 1, one_hot=True)
        feature_a12 = FeatureLabel(5, 10, 1, 1, 1, one_hot=True)
        feature_a13 = FeatureLabel(6, 10, 1, 1, 1, one_hot=True)
        feature_a14 = FeatureLabel(7, 10, 1, 1, 1, one_hot=True)

        self.assertEqual(feature_a01.base_identity, Base.PURINES)
        self.assertEqual(feature_a01.base_index, 1)
        self.assertEqual(feature_a01.base_pair_index, None)
        self.assertEqual(feature_a01.base_hydrogen_bond_count, False)
        self.assertEqual(feature_a01.feature_category, 'A-tract')

        self.assertEqual(feature_a02.base_identity, Base.PYRIMIDINES)
        self.assertEqual(feature_a02.base_index, 1)
        self.assertEqual(feature_a02.base_pair_index, None)
        self.assertEqual(feature_a02.base_hydrogen_bond_count, False)

        self.assertEqual(feature_a03.base_identity, None)
        self.assertEqual(feature_a03.base_index, 1)
        self.assertEqual(feature_a03.base_pair_index, None)
        self.assertEqual(feature_a03.base_hydrogen_bond_count, True)

        self.assertEqual(feature_a04.base_identity, Base.PURINES)
        self.assertEqual(feature_a04.base_index, 2)
        self.assertEqual(feature_a04.base_pair_index, None)
        self.assertEqual(feature_a04.base_hydrogen_bond_count, False)

        self.assertEqual(feature_a05.base_identity, Base.PYRIMIDINES)
        self.assertEqual(feature_a05.base_index, 2)
        self.assertEqual(feature_a05.base_pair_index, None)
        self.assertEqual(feature_a05.base_hydrogen_bond_count, False)

        self.assertEqual(feature_a06.base_identity, None)
        self.assertEqual(feature_a06.base_index, 2)
        self.assertEqual(feature_a06.base_pair_index, None)
        self.assertEqual(feature_a06.base_hydrogen_bond_count, True)

        self.assertEqual(feature_a07.base_identity, Base.A)
        self.assertEqual(feature_a07.base_index, 1)
        self.assertEqual(feature_a07.base_pair_index, None)
        self.assertEqual(feature_a07.base_hydrogen_bond_count, False)

        self.assertEqual(feature_a08.base_identity, Base.C)
        self.assertEqual(feature_a09.base_identity, Base.G)
        self.assertEqual(feature_a10.base_identity, Base.U)
        self.assertEqual(feature_a11.base_identity, Base.A)
        self.assertEqual(feature_a12.base_identity, Base.C)
        self.assertEqual(feature_a13.base_identity, Base.G)
        self.assertEqual(feature_a14.base_identity, Base.U)

        # Loop features

    def test_set_basepair_information(self):
        # Stem features

        feature_s01 = FeatureLabel(30, 10, 10, 1, 1)
        feature_s02 = FeatureLabel(31, 10, 10, 1, 1)
        feature_s03 = FeatureLabel(32, 10, 10, 1, 1)
        feature_s04 = FeatureLabel(33, 10, 10, 1, 1)
        feature_s05 = FeatureLabel(34, 10, 10, 1, 1)
        feature_s06 = FeatureLabel(35, 10, 10, 1, 1)
        feature_s07 = FeatureLabel(36, 10, 10, 1, 1)
        feature_s08 = FeatureLabel(37, 10, 10, 1, 1)

        feature_s09 = FeatureLabel(40, 10, 10, 1, 1, one_hot=True)
        feature_s10 = FeatureLabel(41, 10, 10, 1, 1, one_hot=True)
        feature_s11 = FeatureLabel(42, 10, 10, 1, 1, one_hot=True)
        feature_s12 = FeatureLabel(43, 10, 10, 1, 1, one_hot=True)
        feature_s13 = FeatureLabel(44, 10, 10, 1, 1, one_hot=True)
        feature_s14 = FeatureLabel(45, 10, 10, 1, 1, one_hot=True)
        feature_s15 = FeatureLabel(46, 10, 10, 1, 1, one_hot=True)
        feature_s16 = FeatureLabel(47, 10, 10, 1, 1, one_hot=True)
        feature_s17 = FeatureLabel(48, 10, 10, 1, 1, one_hot=True)

        self.assertEqual(feature_s01.base_identity, Base.PURINES)
        self.assertEqual(feature_s01.base_index, None)
        self.assertEqual(feature_s01.base_pair_index, 1)
        self.assertEqual(feature_s01.stem_shoulder, 'upstream')
        self.assertEqual(feature_s01.base_hydrogen_bond_count, False)
        self.assertEqual(feature_s01.feature_category, 'stem')

        self.assertEqual(feature_s02.base_identity, Base.PYRIMIDINES)
        self.assertEqual(feature_s03.base_identity, None)
        self.assertEqual(feature_s03.base_hydrogen_bond_count, True)

        self.assertEqual(feature_s04.base_identity, Base.PURINES)
        self.assertEqual(feature_s04.stem_shoulder, "downstream")

        self.assertEqual(feature_s05.base_identity, Base.PYRIMIDINES)
        self.assertEqual(feature_s06.base_hydrogen_bond_count, True)

        self.assertEqual(feature_s07.base_pair_index, 1)
        self.assertEqual(feature_s07.stem_shoulder, None)
        self.assertEqual(feature_s07.check_pairing, True)

        self.assertEqual(feature_s08.base_pair_index, 2)
        self.assertEqual(feature_s08.base_identity, Base.PURINES)

        self.assertEqual(feature_s09.base_identity, Base.A)
        self.assertEqual(feature_s10.base_identity, Base.C)
        self.assertEqual(feature_s11.base_identity, Base.G)
        self.assertEqual(feature_s12.base_identity, Base.U)
        self.assertEqual(feature_s13.base_identity, Base.A)
        self.assertEqual(feature_s14.base_identity, Base.C)
        self.assertEqual(feature_s15.base_identity, Base.G)
        self.assertEqual(feature_s16.base_identity, Base.U)
        self.assertEqual(feature_s17.base_identity, None)
        self.assertEqual(feature_s17.check_pairing, True)


if __name__ == '__main__':
    unittest.main()
