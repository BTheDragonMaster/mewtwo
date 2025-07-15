from typing import Optional
from enum import Enum, Flag
from dataclasses import dataclass

from mewtwo.embeddings.sequence import SeqType
from mewtwo.embeddings.bases import Base


class FeatureType(Flag):
    IS_A = 1
    IS_C = 2
    IS_G = 4
    IS_U = 8
    IS_PURINE = 16
    IS_PYRIMIDINE = 32
    NR_H_BONDS = 64
    IS_BONDED = 128
    IS_POT = 256
    IS_BASE_IDENTITY = IS_A | IS_C | IS_G | IS_U
    ONE_HOT_TYPES = IS_BASE_IDENTITY | IS_BONDED | IS_POT
    IS_BASE_FEATURE = IS_PURINE | IS_PYRIMIDINE | NR_H_BONDS
    BASE_FEATURE_TYPES = IS_BASE_FEATURE | IS_BONDED | IS_POT

    @staticmethod
    def from_base(base: Base) -> "FeatureType":
        base_to_feature_type = {Base.A: FeatureType.IS_A,
                                Base.C: FeatureType.IS_C,
                                Base.G: FeatureType.IS_G,
                                Base.U: FeatureType.IS_U,
                                Base.PURINES: FeatureType.IS_PURINE,
                                Base.PYRIMIDINES: FeatureType.IS_PYRIMIDINE}

        assert base in base_to_feature_type

        return base_to_feature_type[base]


class FeatureCategory(Enum):
    A_TRACT = 1
    STEM = 2
    LOOP = 3
    U_TRACT = 4


class StemShoulder(Enum):
    UPSTREAM = 1
    DOWNSTREAM = 2


@dataclass
class FeatureLabel:
    feature_type: FeatureType
    feature_category: FeatureCategory
    base_index: Optional[int] = None
    stem_shoulder: Optional[StemShoulder] = None

    def __hash__(self):
        return hash((self.feature_type, self.feature_category, self.base_index, self.stem_shoulder))

    def __repr__(self):

        if self.feature_category == FeatureCategory.STEM:
            base_description = f"basepair_{self.base_index}"
            if self.feature_type != FeatureType.IS_BONDED:
                base_description = f"{base_description}_{self.stem_shoulder.name}"
        else:
            base_description = f"base_{self.base_index}"

        return f"{self.feature_category.name}|{base_description}|{self.feature_type.name}"

    @classmethod
    def from_string(cls, feature_string: str) -> "FeatureLabel":
        category, base_information, feature_type = feature_string.split('|')
        feature_category = FeatureCategory[category]
        base_info_split = base_information.split('_')[1:]
        base_index = int(base_info_split[0])
        feature_type = FeatureType[feature_type.strip()]

        shoulder = None
        if feature_category == FeatureCategory.STEM and feature_type != FeatureType.IS_BONDED:
            assert len(base_info_split) == 2
            shoulder = StemShoulder[base_info_split[1]]

        return cls(feature_type, feature_category, base_index, shoulder)

    @classmethod
    def from_feature_position(cls, feature_position: int, max_a_tract_length: int,
                              max_stem_length: int, max_loop_length: int,
                              max_u_tract_length: int, seq_type: SeqType = SeqType.RNA,
                              one_hot: bool = False, utract_has_pot: bool = True) -> "FeatureLabel":

        if one_hot:
            nr_a_tract_features = 4 * max_a_tract_length
            nr_stem_features = 9 * max_stem_length
            nr_loop_features = 4 * max_loop_length

            if utract_has_pot:
                nr_u_tract_features = 5 * max_u_tract_length
            else:
                nr_u_tract_features = 4 * max_u_tract_length

        else:
            nr_a_tract_features = 3 * max_a_tract_length
            nr_stem_features = 7 * max_stem_length
            nr_loop_features = 3 * max_loop_length

            if utract_has_pot:
                nr_u_tract_features = 4 * max_u_tract_length
            else:
                nr_u_tract_features = 3 * max_u_tract_length

        if feature_position < nr_a_tract_features:
            feature_category = FeatureCategory.A_TRACT
            relative_feature_position = feature_position

        elif feature_position < nr_a_tract_features + nr_stem_features:
            feature_category = FeatureCategory.STEM
            relative_feature_position = feature_position - nr_a_tract_features

        elif feature_position < nr_a_tract_features + nr_stem_features + nr_loop_features:
            feature_category = FeatureCategory.LOOP
            relative_feature_position = feature_position - nr_a_tract_features - nr_stem_features

        elif feature_position < nr_a_tract_features + nr_stem_features + nr_loop_features + nr_u_tract_features:
            feature_category = FeatureCategory.U_TRACT
            relative_feature_position = feature_position - nr_a_tract_features - nr_stem_features - nr_loop_features

        else:
            raise IndexError(f"Feature position {feature_position} does not exist.")

        if feature_category != FeatureCategory.STEM:
            feature_type, base_index = cls.get_base_information(one_hot, seq_type, feature_category,
                                                                relative_feature_position, utract_has_pot)
            shoulder = None
        else:
            feature_type, base_index, shoulder = cls.get_basepair_information(one_hot, seq_type,
                                                                              relative_feature_position)

        return cls(feature_type, feature_category, base_index, shoulder)

    @staticmethod
    def get_base_information(one_hot: bool, seq_type: SeqType, feature_category: FeatureCategory,
                             relative_feature_position: int,
                             utract_has_pot: bool) -> tuple[FeatureType, Optional[int]]:

        if one_hot:
            if seq_type == SeqType.RNA:
                features = [Base.A, Base.C, Base.G, Base.U]
            elif seq_type == SeqType.DNA:
                features = [Base.A, Base.C, Base.G, Base.T]
            else:
                raise ValueError(f"Unsupported sequence type: {seq_type}")

        else:
            features = [Base.PURINES, Base.PYRIMIDINES, "hydrogen bonds"]

        if feature_category == FeatureCategory.U_TRACT and utract_has_pot:
            features.append('POT')

        feature_nr = len(features)

        for i in range(feature_nr):
            if relative_feature_position % feature_nr == i:
                feature = features[i]
                if feature == 'POT':
                    feature_type = FeatureType.IS_POT
                elif feature == 'hydrogen bonds':
                    feature_type = FeatureType.NR_H_BONDS
                else:
                    feature_type = FeatureType.from_base(feature)

                base_index = relative_feature_position // feature_nr + 1
                break

        else:
            raise ValueError(f"Could not find feature type for feature at {relative_feature_position}")

        return feature_type, base_index

    @staticmethod
    def get_basepair_information(one_hot: bool, seq_type: SeqType,
                                 relative_feature_position: int) -> tuple[FeatureType, Optional[int], Optional[str]]:
        if one_hot:
            feature_nr = 9
            if seq_type == SeqType.RNA:
                features = [Base.A, Base.C, Base.G, Base.U, Base.A, Base.C, Base.G, Base.U, 'bonded']
            elif seq_type == SeqType.DNA:
                features = [Base.A, Base.C, Base.G, Base.T, Base.A, Base.C, Base.G, Base.T, 'bonded']
            else:
                raise ValueError(f"Unsupported sequence type: {seq_type}")

        else:
            feature_nr = 7
            features = [Base.PURINES, Base.PYRIMIDINES, 'hydrogen bonds',
                        Base.PURINES, Base.PYRIMIDINES, 'hydrogen bonds', 'bonded']

        stem_shoulder: Optional[StemShoulder] = None

        for i in range(feature_nr):
            if relative_feature_position % feature_nr == i:
                feature = features[i]
                base_index = relative_feature_position // feature_nr + 1

                if feature != 'bonded':
                    if i < len(features) // 2:
                        stem_shoulder = StemShoulder.UPSTREAM

                    else:
                        stem_shoulder = StemShoulder.DOWNSTREAM

                    if feature != 'hydrogen bonds':
                        feature_type = FeatureType.from_base(feature)
                    else:
                        feature_type = FeatureType.NR_H_BONDS

                else:
                    feature_type = FeatureType.IS_BONDED
                break
        else:

            raise ValueError(f"Could not find feature type for feature at {relative_feature_position}")

        return feature_type, base_index, stem_shoulder
