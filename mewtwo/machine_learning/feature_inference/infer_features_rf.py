import os
from statistics import mean, stdev

from mewtwo.parsers.parse_feature_file import parse_feature_file
from mewtwo.data_processing.iterate_over_dir import iterate_over_dir
from mewtwo.writers.write_feature_importances import write_feature_importances


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

