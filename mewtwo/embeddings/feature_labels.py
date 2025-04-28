from typing import Optional

from mewtwo.embeddings.sequence import SeqType
from mewtwo.embeddings.bases import Base


class FeatureLabel:

    def __init__(self, feature_position: int, max_a_tract_length: int,
                 max_stem_length: int, max_loop_length: int,
                 max_u_tract_length: int, seq_type: SeqType = SeqType.RNA,
                 one_hot: bool = False) -> None:

        self.seq_type = seq_type
        self.one_hot = one_hot

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
            self.feature_type = "A-tract"
            self.relative_feature_position = feature_position
            self.set_base_information()
        elif feature_position < nr_a_tract_features + nr_stem_features:
            self.feature_type = "stem"
            self.relative_feature_position = feature_position - nr_a_tract_features
            self.set_basepair_information()
        elif feature_position < nr_a_tract_features + nr_stem_features + nr_loop_features:
            self.feature_type = "loop"
            self.relative_feature_position = feature_position - nr_a_tract_features - nr_stem_features
            self.set_base_information()
        elif feature_position < nr_a_tract_features + nr_stem_features + nr_loop_features + nr_u_tract_features:
            self.feature_type = "U-tract"
            self.relative_feature_position = \
                feature_position - nr_a_tract_features - nr_stem_features - nr_loop_features
            self.set_base_information()
        else:
            raise IndexError(f"Feature position {feature_position} does not exist.")

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

        if self.feature_type == 'U-tract':
            features.append('POT')

        feature_nr = len(features)

        for i in range(feature_nr):
            if self.relative_feature_position % feature_nr == i:
                feature = features[i]
                if feature == 'POT':
                    self.check_pot = True
                elif feature == 'hydrogen bonds':
                    self.base_hydrogen_bond_count = True
                else:
                    self.base_identity = feature

                self.base_index = self.relative_feature_position // feature_nr + 1
                break

        else:
            raise ValueError(f"Could not find feature type for feature at {self.relative_feature_position}")

    def set_basepair_information(self) -> None:
        assert self.feature_type == "stem"
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
                    else:
                        self.base_hydrogen_bond_count = True

                else:
                    self.check_pairing = True
                break
        else:

            raise ValueError(f"Could not find feature type for feature at {self.relative_feature_position}")
