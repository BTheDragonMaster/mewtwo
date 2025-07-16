from sklearn.neural_network import MLPRegressor

from mewtwo.machine_learning.prepare_data import terminators_to_ml_input


def train_nn(train_terminators, test_terminators, one_hot=True):
    train_x, train_y, test_x, test_y = terminators_to_ml_input(train_terminators, test_terminators, one_hot=one_hot)
    print(len(train_x))
    nn = MLPRegressor(hidden_layer_sizes=(500, 1000, 1000, 500, 100), random_state=100125, max_iter=1000, solver='adam')
    nn.fit(train_x, train_y)

    print(nn.score(test_x, test_y))