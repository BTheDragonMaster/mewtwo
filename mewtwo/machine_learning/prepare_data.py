from mewtwo.embeddings.terminator.terminator import get_terminator_part_sizes, Terminator


def terminators_to_ml_input(train_terminators: list[Terminator], test_terminators: list[Terminator], one_hot=True) -> \
        tuple[list[list[int]], list[float], list[list[int]], list[float]]:
    max_loop, max_stem, max_a, max_u = get_terminator_part_sizes(train_terminators + test_terminators)

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

    return train_x, train_y, test_x, test_y
