from mewtwo.parsers.tabular import Tabular


def parse_feature_file(input_file: str) -> dict[str, float]:

    feature_to_importance = {}
    feature_data = Tabular(input_file, [0])
    for feature in feature_data.data:

        feature_name = feature_data.get_value(feature, "feature_name")
        importance = float(feature_data.get_value(feature, "feature_importance"))
        feature_to_importance[feature_name] = importance

    return feature_to_importance
