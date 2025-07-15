import os
from statistics import mean, stdev

from mewtwo.parsers.parse_feature_file import parse_feature_file
from mewtwo.data_processing.iterate_over_dir import iterate_over_dir
from mewtwo.writers.write_feature_importances import write_feature_importances
from mewtwo.embeddings.feature_labels import FeatureCategory, FeatureLabel, FeatureType, StemShoulder


def get_average_features_from_crossvalidation(input_folder, get_stdev=True):
    feature_to_importance_list = {}
    feature_to_stdev = {}
    for folder_name, folder_path in iterate_over_dir(input_folder, get_dirs=True):
        if 'crossvalidation_results' in folder_name:
            feature_file = os.path.join(folder_path, "feature_importances.txt")
            feature_to_importance = parse_feature_file(feature_file)
            for feature, importance in feature_to_importance.items():
                if feature not in feature_to_importance_list:
                    feature_to_importance_list[feature] = []

                feature_to_importance_list[feature].append(importance)
    feature_to_average_importance = {}

    for feature, importances in feature_to_importance_list.items():
        if not importances:
            raise ValueError(f"No importance values found for feature {feature}")
        elif get_stdev and len(importances) < 2:
            raise ValueError(f"Too few importance values for feature {feature} to calculate stdev")
        feature_to_average_importance[feature] = mean(importances)
        if get_stdev:
            feature_to_stdev[feature] = stdev(importances)

    if get_stdev:
        return feature_to_average_importance, feature_to_stdev

    else:
        return feature_to_average_importance


def write_average_importances(input_folder: str, out_file: str) -> None:
    feature_to_average_importance, feature_to_stdev = get_average_features_from_crossvalidation(input_folder)
    write_feature_importances(feature_to_average_importance, out_file, feature_to_stdev=feature_to_stdev,
                              sort_by_importance=True)


class SortedFeatures:
    def __init__(self, summed_feature_to_importance):
        sorted_features = {StemShoulder.UPSTREAM: [],
                           StemShoulder.DOWNSTREAM: [],
                           FeatureCategory.U_TRACT: [],
                           FeatureCategory.A_TRACT: [],
                           FeatureCategory.LOOP: []}
        for summed_feature, importance in summed_feature_to_importance.items():
            if summed_feature.feature_category != FeatureCategory.STEM:
                sorted_features[summed_feature.feature_category].append((summed_feature, importance))

            else:
                sorted_features[summed_feature.stem_shoulder].append((summed_feature, importance))

        self.a_tract = sorted(sorted_features[FeatureCategory.A_TRACT], key=lambda x: x[0].base_index)
        self.u_tract = sorted(sorted_features[FeatureCategory.U_TRACT], key=lambda x: x[0].base_index)
        self.loop = sorted(sorted_features[FeatureCategory.LOOP], key=lambda x: x[0].base_index)
        self.left_stem_shoulder = sorted(sorted_features[StemShoulder.UPSTREAM], key=lambda x: x[0].base_index)
        self.right_stem_shoulder = sorted(sorted_features[StemShoulder.DOWNSTREAM], key=lambda x: x[0].base_index)


def sum_importance_per_base(feature_to_importance):
    importance_dict = {StemShoulder.UPSTREAM: {},
                       StemShoulder.DOWNSTREAM: {},
                       FeatureCategory.U_TRACT: {},
                       FeatureCategory.A_TRACT: {},
                       FeatureCategory.LOOP: {}}

    one_hot = False

    for feature, importance in feature_to_importance.items():
        if feature.feature_type in FeatureType.IS_BASE_IDENTITY:
            one_hot = True

        if feature.feature_category != FeatureCategory.STEM:
            if feature.base_index not in importance_dict[feature.feature_category]:
                importance_dict[feature.feature_category][feature.base_index] = []
            importance_dict[feature.feature_category][feature.base_index].append(importance)
        else:
            if feature.base_index not in importance_dict[StemShoulder.UPSTREAM]:
                importance_dict[StemShoulder.UPSTREAM][feature.base_index] = []
            if feature.base_index not in importance_dict[StemShoulder.DOWNSTREAM]:
                importance_dict[StemShoulder.DOWNSTREAM][feature.base_index] = []

            if feature.stem_shoulder is None:
                importance_dict[StemShoulder.UPSTREAM][feature.base_index].append(importance)
                importance_dict[StemShoulder.DOWNSTREAM][feature.base_index].append(importance)
            elif feature.stem_shoulder == StemShoulder.UPSTREAM:
                importance_dict[StemShoulder.UPSTREAM][feature.base_index].append(importance)
            elif feature.stem_shoulder == StemShoulder.DOWNSTREAM:
                importance_dict[StemShoulder.DOWNSTREAM][feature.base_index].append(importance)
            else:
                raise ValueError(f"Unknown value for stem shoulder: {feature.stem_shoulder}")

    summed_importances = {}

    for category, base_to_importances in importance_dict.items():
        if type(category) == StemShoulder:
            feature_category = FeatureCategory.STEM
            stem_shoulder = category
        else:
            feature_category = category
            stem_shoulder = None

        for base, importances in base_to_importances.items():

            if one_hot:
                feature_type = FeatureType.ONE_HOT_TYPES
            else:
                feature_type = FeatureType.BASE_FEATURE_TYPES

            feature = FeatureLabel(feature_type, feature_category, base, stem_shoulder)
            summed_importances[feature] = sum(importances)

    return summed_importances


def get_normalized_importances(feature_file):
    feature_to_importance = parse_feature_file(feature_file)
    normalize_importances(feature_to_importance)

    return feature_to_importance


def normalize_importances(feature_to_importance):
    max_importance = max(feature_to_importance.values())
    factor = 1.0 / max_importance

    for feature in feature_to_importance:
        feature_to_importance[feature] = factor * feature_to_importance[feature]

