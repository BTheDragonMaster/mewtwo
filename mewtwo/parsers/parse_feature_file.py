from mewtwo.parsers.tabular import Tabular
from mewtwo.embeddings.feature_labels import FeatureLabel
from mewtwo.embeddings.sequence import SeqType


def parse_feature_file(input_file: str) -> dict[FeatureLabel, float]:

    feature_to_importance = {}
    feature_data = Tabular(input_file, [0])
    for feature in feature_data.data:

        feature_name = feature_data.get_value(feature, "feature_name")
        importance = float(feature_data.get_value(feature, "feature_importance"))
        feature_to_importance[FeatureLabel.from_string(feature_name)] = importance

    return feature_to_importance
