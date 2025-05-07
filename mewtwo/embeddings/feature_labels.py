from typing import Optional

from mewtwo.embeddings.sequence import SeqType
from mewtwo.embeddings.bases import Base
from enum import Enum

class FeatureType(Enum):
    IS_A = 1
    IS_C = 2
    IS_G = 3
    IS_U = 4
    IS_PURINE = 5
    IS_PYRIMIDINE = 6
    NR_H_BONDS = 7
    IS_BONDED = 8
    IS_POT = 9

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


class FeatureLabel:

    def __init__(self, feature_position: int, max_a_tract_length: int,
                 max_stem_length: int, max_loop_length: int,
                 max_u_tract_length: int, seq_type: SeqType = SeqType.RNA,
                 one_hot: bool = False) -> None:

        self.seq_type = seq_type
        self.one_hot = one_hot

        self.feature_type: Optional[FeatureType] = None

        self.base_index: Optional[int] = None
        self.base_identity: Optional[Base] = None
        self.base_hydrogen_bond_count: bool = False

        self.check_pot: bool = False

        self.base_pair_index: Optional[int] = None
        self.stem_shoulder: Optional[str] = None
        self.check_pairing: bool = False

        if self.one_hot:
            nr_a_tract_features = 4 * max_a_tract_length
            nr_stem_features = 9 * max_stem_length
            nr_loop_features = 4 * max_loop_length

            # Assumes separate feature for POT
            nr_u_tract_features = 5 * max_u_tract_length

        else:
            nr_a_tract_features = 3 * max_a_tract_length
            nr_stem_features = 7 * max_stem_length
            nr_loop_features = 3 * max_loop_length

            # Assumes separate feature for POT
            nr_u_tract_features = 4 * max_u_tract_length

        if feature_position < nr_a_tract_features:
            self.feature_category = "A-tract"
            self.relative_feature_position = feature_position
            self.set_base_information()
        elif feature_position < nr_a_tract_features + nr_stem_features:
            self.feature_category = "stem"
            self.relative_feature_position = feature_position - nr_a_tract_features
            self.set_basepair_information()
        elif feature_position < nr_a_tract_features + nr_stem_features + nr_loop_features:
            self.feature_category = "loop"
            self.relative_feature_position = feature_position - nr_a_tract_features - nr_stem_features
            self.set_base_information()
        elif feature_position < nr_a_tract_features + nr_stem_features + nr_loop_features + nr_u_tract_features:
            self.feature_category = "U-tract"
            self.relative_feature_position = \
                feature_position - nr_a_tract_features - nr_stem_features - nr_loop_features
            self.set_base_information()
        else:
            raise IndexError(f"Feature position {feature_position} does not exist.")

    def __str__(self):
        pass

    def set_base_information(self) -> None:

        if self.one_hot:
            if self.seq_type == SeqType.RNA:
                features = [Base.A, Base.C, Base.G, Base.U]
            elif self.seq_type == SeqType.DNA:
                features = [Base.A, Base.C, Base.G, Base.T]
            else:
                raise ValueError(f"Unsupported sequence type: {self.seq_type}")

        else:
            features = [Base.PURINES, Base.PYRIMIDINES, "hydrogen bonds"]

        if self.feature_category == 'U-tract':
            features.append('POT')

        feature_nr = len(features)

        for i in range(feature_nr):
            if self.relative_feature_position % feature_nr == i:
                feature = features[i]
                if feature == 'POT':
                    self.check_pot = True
                    self.feature_type = FeatureType.IS_POT
                elif feature == 'hydrogen bonds':
                    self.base_hydrogen_bond_count = True
                    self.feature_type = FeatureType.NR_H_BONDS
                else:
                    self.base_identity = feature
                    self.feature_type = FeatureType.from_base(feature)

                self.base_index = self.relative_feature_position // feature_nr + 1
                break

        else:
            raise ValueError(f"Could not find feature type for feature at {self.relative_feature_position}")

    def set_basepair_information(self) -> None:
        assert self.feature_category == "stem"
        if self.one_hot:
            feature_nr = 9
            if self.seq_type == SeqType.RNA:
                features = [Base.A, Base.C, Base.G, Base.U, Base.A, Base.C, Base.G, Base.U, 'bonded']
            elif self.seq_type == SeqType.DNA:
                features = [Base.A, Base.C, Base.G, Base.T, Base.A, Base.C, Base.G, Base.T, 'bonded']
            else:
                raise ValueError(f"Unsupported sequence type: {self.seq_type}")

        else:
            feature_nr = 7
            features = [Base.PURINES, Base.PYRIMIDINES, 'hydrogen bonds',
                        Base.PURINES, Base.PYRIMIDINES, 'hydrogen bonds', 'bonded']

        for i in range(feature_nr):
            if self.relative_feature_position % feature_nr == i:
                feature = features[i]
                self.base_pair_index = self.relative_feature_position // feature_nr + 1

                if feature != 'bonded':
                    if i < len(features) // 2:
                        self.stem_shoulder = 'upstream'

                    else:
                        self.stem_shoulder = 'downstream'

                    if feature != 'hydrogen bonds':
                        self.base_identity = feature
                        self.feature_type = FeatureType.from_base(feature)
                    else:
                        self.base_hydrogen_bond_count = True
                        self.feature_type = FeatureType.NR_H_BONDS

                else:
                    self.check_pairing = True
                    self.feature_type = FeatureType.IS_BONDED
                break
        else:

            raise ValueError(f"Could not find feature type for feature at {self.relative_feature_position}")
