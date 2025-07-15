import os
from argparse import ArgumentParser, Namespace
from enum import Enum

from joblib import dump

from mewtwo.parsers.parse_data_chen import get_chen_terminators
from mewtwo.machine_learning.data_preparation.train_test_split import split_data
from mewtwo.machine_learning.random_forest.train_random_forest import train_random_forest, RandomForestMode, \
    FeaturisationMode
from mewtwo.machine_learning.feature_inference.infer_features_rf import write_average_importances
from mewtwo.parsers.parse_termite_output import get_termite_terminators
from mewtwo.embeddings.terminator.draw_terminator import visualise_feature_importances


class DataSource(Enum):
    TERMITE = 1
    CHEN = 2


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Train random forest from E. coli terminator data published by Chen et al.")
    parser.add_argument("-i", type=str, required=True, help="Path to input file.")
    parser.add_argument("-o", type=str, required=True, help="Path to output directory.")
    parser.add_argument("-d", type=str, default="CHEN", help="Data source. Must be CHEN or TERMITE")
    parser.add_argument("-m", type=str, default="FULL", help="Random forest training mode. Must be one of \
    'FULL', 'TRAIN', or 'CROSSVALIDATION'.")
    parser.add_argument("-a", type=str, default="is_synthetic", help="Attribute for stratified data splitting")
    parser.add_argument("-f", type=str, default="ONE_HOT", help="Featurisation mode. Must be one of 'ONE_HOT' or \
    'PURINE_PYRIMIDINE'.")
    parser.add_argument("-s", action="store_true", help="If given, save random forest models.")
    parser.add_argument("-n", type=int, default=100, help="Number of trees in RF")

    args = parser.parse_args()
    return args


def rf_from_data(data_file: str, data_source: DataSource, out_dir: str, attribute: str, save_model: bool = False,
                 mode: RandomForestMode = RandomForestMode.CROSSVALIDATION,
                 featurisation_mode: FeaturisationMode = FeaturisationMode.ONE_HOT, n_trees: int = 100) -> None:
    if data_source == DataSource.CHEN:
        terminators = get_chen_terminators(data_file)
    elif data_source == DataSource.TERMITE:
        terminators = get_termite_terminators(data_file, species_column=True)
    else:
        raise ValueError(f"Unknown data source: {data_source.name}")

    train_terminators, test_terminators, crossvalidation_sets = split_data(terminators,
                                                                           attribute_for_splitting=attribute)

    if featurisation_mode == FeaturisationMode.ONE_HOT:
        one_hot = True
    else:
        one_hot = False

    if RandomForestMode.CROSSVALIDATION in mode:
        figure_dir = os.path.join(out_dir, "feature_importance_visualisations_crossval")

        for crossval_nr, crossvalidation_set in crossvalidation_sets.items():
            out_path = os.path.join(out_dir, f"crossvalidation_results_{crossval_nr}")

            rf = train_random_forest(crossvalidation_set.train, crossvalidation_set.test, one_hot=one_hot,
                                     out_dir=out_path, n_trees=n_trees)
            if save_model:
                model_path = os.path.join(out_dir, f"crossvalidation_model_{crossval_nr}.rf")
                dump(rf, model_path)

        averaged_features_dir = os.path.join(out_dir, "average_feature_importances.txt")

        write_average_importances(out_dir, averaged_features_dir)
        visualise_feature_importances(averaged_features_dir, figure_dir)

    if RandomForestMode.TRAIN in mode:
        figure_dir = os.path.join(out_dir, "feature_importance_visualisations_train")

        rf = train_random_forest(train_terminators, test_terminators, one_hot=one_hot, out_dir=out_dir, n_trees=n_trees)
        if save_model:
            model_path = os.path.join(out_dir, f"predictor.rf")
            dump(rf, model_path)

        visualise_feature_importances(os.path.join(out_dir, "feature_importances.txt"), figure_dir)


def main():
    args = parse_arguments()
    if not os.path.exists(args.o):
        os.mkdir(args.o)

    rf_from_data(args.i, DataSource[args.d], args.o, args.a, args.s, RandomForestMode[args.m],
                 FeaturisationMode[args.f], args.n)


if __name__ == "__main__":
    main()
