from sklearn.ensemble import RandomForestRegressor

from mewtwo.embeddings.terminator.terminator import get_terminator_part_sizes
from mewtwo.embeddings.feature_labels import FeatureLabel


def train_random_forest(train_terminators, test_terminators, one_hot=False):
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

    random_forest = RandomForestRegressor(n_estimators=1000, oob_score=True)
    random_forest.fit(train_x, train_y)

    print("oob", random_forest.oob_score_)
    importance_and_label = []
    for i, feature_importance in enumerate(random_forest.feature_importances_):
        feature_label = FeatureLabel(i, max_a, max_stem, max_loop, max_u, one_hot=one_hot)
        importance_and_label.append((feature_importance, feature_label))

    importance_and_label.sort(key=lambda x: x[0], reverse=True)

    for importance, label in importance_and_label[:20]:
        if label.feature_category == 'stem':
            print(label.feature_category, label.feature_type, importance, label.base_pair_index, label.base_identity, label.stem_shoulder)
        else:
            print(label.feature_category, label.feature_type, importance, label.base_index, label.base_identity)

    print("test score", random_forest.score(test_x, test_y))

    return random_forest



