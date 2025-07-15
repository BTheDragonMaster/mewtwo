import unittest
from mewtwo.embeddings.feature_labels import FeatureLabel, FeatureCategory, FeatureType, StemShoulder


class TestFeatureLabel(unittest.TestCase):

    def test_set_base_information(self):

        # A-tract features

        feature_a01 = FeatureLabel.from_feature_position(0, 10, 1, 1, 1)
        feature_a02 = FeatureLabel.from_feature_position(1, 10, 1, 1, 1)
        feature_a03 = FeatureLabel.from_feature_position(2, 10, 1, 1, 1)
        feature_a04 = FeatureLabel.from_feature_position(3, 10, 1, 1, 1)
        feature_a05 = FeatureLabel.from_feature_position(4, 10, 1, 1, 1)
        feature_a06 = FeatureLabel.from_feature_position(5, 10, 1, 1, 1)

        feature_a07 = FeatureLabel.from_feature_position(0, 10, 1, 1, 1, one_hot=True)
        feature_a08 = FeatureLabel.from_feature_position(1, 10, 1, 1, 1, one_hot=True)
        feature_a09 = FeatureLabel.from_feature_position(2, 10, 1, 1, 1, one_hot=True)
        feature_a10 = FeatureLabel.from_feature_position(3, 10, 1, 1, 1, one_hot=True)
        feature_a11 = FeatureLabel.from_feature_position(4, 10, 1, 1, 1, one_hot=True)
        feature_a12 = FeatureLabel.from_feature_position(5, 10, 1, 1, 1, one_hot=True)
        feature_a13 = FeatureLabel.from_feature_position(6, 10, 1, 1, 1, one_hot=True)
        feature_a14 = FeatureLabel.from_feature_position(7, 10, 1, 1, 1, one_hot=True)

        self.assertEqual(feature_a01.feature_type, FeatureType.IS_PURINE)
        self.assertEqual(feature_a01.base_index, 1)
        self.assertEqual(feature_a01.stem_shoulder, None)
        self.assertEqual(feature_a01.feature_category, FeatureCategory.A_TRACT)

        self.assertEqual(feature_a02.feature_type, FeatureType.IS_PYRIMIDINE)
        self.assertEqual(feature_a02.base_index, 1)

        self.assertEqual(feature_a03.base_index, 1)
        self.assertEqual(feature_a03.feature_type, FeatureType.NR_H_BONDS)

        self.assertEqual(feature_a04.feature_type, FeatureType.IS_PURINE)
        self.assertEqual(feature_a04.base_index, 2)

        self.assertEqual(feature_a05.feature_type, FeatureType.IS_PYRIMIDINE)
        self.assertEqual(feature_a05.base_index, 2)

        self.assertEqual(feature_a06.base_index, 2)
        self.assertEqual(feature_a06.feature_type, FeatureType.NR_H_BONDS)

        self.assertEqual(feature_a07.feature_type, FeatureType.IS_A)
        self.assertEqual(feature_a07.base_index, 1)

        self.assertEqual(feature_a08.feature_type, FeatureType.IS_C)
        self.assertEqual(feature_a09.feature_type, FeatureType.IS_G)
        self.assertEqual(feature_a10.feature_type, FeatureType.IS_U)
        self.assertEqual(feature_a11.feature_type, FeatureType.IS_A)
        self.assertEqual(feature_a12.feature_type, FeatureType.IS_C)
        self.assertEqual(feature_a13.feature_type, FeatureType.IS_G)
        self.assertEqual(feature_a14.feature_type, FeatureType.IS_U)

        # Loop features

    def test_set_basepair_information(self):
        # Stem features

        feature_s01 = FeatureLabel.from_feature_position(30, 10, 10, 1, 1)
        feature_s02 = FeatureLabel.from_feature_position(31, 10, 10, 1, 1)
        feature_s03 = FeatureLabel.from_feature_position(32, 10, 10, 1, 1)
        feature_s04 = FeatureLabel.from_feature_position(33, 10, 10, 1, 1)
        feature_s05 = FeatureLabel.from_feature_position(34, 10, 10, 1, 1)
        feature_s06 = FeatureLabel.from_feature_position(35, 10, 10, 1, 1)
        feature_s07 = FeatureLabel.from_feature_position(36, 10, 10, 1, 1)
        feature_s08 = FeatureLabel.from_feature_position(37, 10, 10, 1, 1)

        feature_s09 = FeatureLabel.from_feature_position(40, 10, 10, 1, 1, one_hot=True)
        feature_s10 = FeatureLabel.from_feature_position(41, 10, 10, 1, 1, one_hot=True)
        feature_s11 = FeatureLabel.from_feature_position(42, 10, 10, 1, 1, one_hot=True)
        feature_s12 = FeatureLabel.from_feature_position(43, 10, 10, 1, 1, one_hot=True)
        feature_s13 = FeatureLabel.from_feature_position(44, 10, 10, 1, 1, one_hot=True)
        feature_s14 = FeatureLabel.from_feature_position(45, 10, 10, 1, 1, one_hot=True)
        feature_s15 = FeatureLabel.from_feature_position(46, 10, 10, 1, 1, one_hot=True)
        feature_s16 = FeatureLabel.from_feature_position(47, 10, 10, 1, 1, one_hot=True)
        feature_s17 = FeatureLabel.from_feature_position(48, 10, 10, 1, 1, one_hot=True)

        self.assertEqual(feature_s01.feature_type, FeatureType.IS_PURINE)
        self.assertEqual(feature_s01.base_index, 1)

        self.assertEqual(feature_s01.stem_shoulder, StemShoulder.UPSTREAM)
        self.assertEqual(feature_s01.feature_category, FeatureCategory.STEM)

        self.assertEqual(feature_s02.feature_type, FeatureType.IS_PYRIMIDINE)

        self.assertEqual(feature_s03.feature_type, FeatureType.NR_H_BONDS)
        self.assertEqual(feature_s03.feature_category, FeatureCategory.STEM)

        self.assertEqual(feature_s04.feature_type, FeatureType.IS_PURINE)
        self.assertEqual(feature_s04.stem_shoulder, StemShoulder.DOWNSTREAM)

        self.assertEqual(feature_s05.feature_type, FeatureType.IS_PYRIMIDINE)
        self.assertEqual(feature_s06.feature_type, FeatureType.NR_H_BONDS)

        self.assertEqual(feature_s07.base_index, 1)
        self.assertEqual(feature_s07.stem_shoulder, None)

        self.assertEqual(feature_s08.base_index, 2)
        self.assertEqual(feature_s08.feature_type, FeatureType.IS_PURINE)

        self.assertEqual(feature_s09.feature_type, FeatureType.IS_A)
        self.assertEqual(feature_s10.feature_type, FeatureType.IS_C)
        self.assertEqual(feature_s11.feature_type, FeatureType.IS_G)
        self.assertEqual(feature_s12.feature_type, FeatureType.IS_U)
        self.assertEqual(feature_s13.feature_type, FeatureType.IS_A)
        self.assertEqual(feature_s14.feature_type, FeatureType.IS_C)
        self.assertEqual(feature_s15.feature_type, FeatureType.IS_G)
        self.assertEqual(feature_s16.feature_type, FeatureType.IS_U)
        self.assertEqual(feature_s17.feature_type, FeatureType.IS_BONDED)


if __name__ == '__main__':
    unittest.main()
