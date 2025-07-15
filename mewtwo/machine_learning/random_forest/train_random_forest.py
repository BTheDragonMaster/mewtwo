import os
from enum import Enum, Flag
from typing import Optional

from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr

from mewtwo.embeddings.terminator.terminator import get_terminator_part_sizes, Terminator
from mewtwo.embeddings.feature_labels import FeatureLabel, FeatureCategory


class FeaturisationMode(Enum):
    ONE_HOT = 1
    PURINE_PYRIMIDINE = 2


class RandomForestMode(Flag):
    CROSSVALIDATION = 1
    TRAIN = 2
    FULL = CROSSVALIDATION | TRAIN


def train_random_forest(train_terminators: list[Terminator], test_terminators: list[Terminator],
                        out_dir: Optional[str] = None, one_hot: bool = True, n_trees: int = 100):

    features_out = os.path.join(out_dir, "feature_importances.txt")
    performance_out = os.path.join(out_dir, "performance.txt")
    datapoints_out = os.path.join(out_dir, "actual_vs_predicted.txt")

    all_terminators = train_terminators + test_terminators
    max_loop, max_stem, max_a, max_u = get_terminator_part_sizes(all_terminators)
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for terminator in train_terminators:
        train_x.append(terminator.to_vector(max_loop, max_stem, max_a, max_u, one_hot=one_hot))
        train_y.append(terminator.te)

    for terminator in test_terminators:
        test_x.append(terminator.to_vector(max_loop, max_stem, max_a, max_u, one_hot=one_hot))
        test_y.append(terminator.te)

    random_forest = RandomForestRegressor(n_estimators=n_trees, oob_score=True)
    random_forest.fit(train_x, train_y)

    print("oob", random_forest.oob_score_)
    importance_and_label = []
    if train_terminators[0].u_tract.pot is not None:
        u_tract_pot = True
    else:
        u_tract_pot = False
    for i, feature_importance in enumerate(random_forest.feature_importances_):
        feature_label = FeatureLabel.from_feature_position(i, max_a, max_stem, max_loop, max_u, one_hot=one_hot,
                                                           utract_has_pot=u_tract_pot)
        importance_and_label.append((feature_importance, feature_label))

    importance_and_label.sort(key=lambda x: x[0], reverse=True)

    for importance, label in importance_and_label[:20]:
        if label.feature_category == FeatureCategory.STEM:
            print(label.feature_category, label.feature_type, importance, label.base_index, label.stem_shoulder)
        else:
            print(label.feature_category, label.feature_type, importance, label.base_index)

    print("test score", random_forest.score(test_x, test_y))
    print("Pearson correlation: ", pearsonr(random_forest.predict(test_x), test_y))

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        with open(features_out, 'w') as features:
            features.write("feature_name\tfeature_importance\n")
            for importance, label in importance_and_label:

                features.write(f"{label}\t{importance:.10f}\n")

        with open(performance_out, 'w') as out:
            out.write("test_score\tpearson\tspearman\n")
            out.write(f"{random_forest.score(test_x, test_y):.10f}\t{pearsonr(random_forest.predict(test_x), test_y).statistic:.10f}\t{spearmanr(random_forest.predict(test_x), test_y).statistic}")

        with open(datapoints_out, 'w') as out:
            out.write(f"actual\tpredicted\n")
            for i, prediction in enumerate(random_forest.predict(test_x)):
                true_label = test_y[i]
                out.write(f"{true_label}\t{prediction}\n")
    return random_forest



