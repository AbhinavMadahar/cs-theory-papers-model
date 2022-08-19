from multiprocessing import Pool, cpu_count
from typing import Callable, Iterable, Tuple


def beam_search(criterion, children, seed_configurations) -> None:
    """
    Implements a beam search.

    Arguments:
        criterion: a function which accepts a hyperparameter configuration and returns the configuration and a score representing how good it is. 
                   the score can be of any type as long as it implements __gt__.
        children: a function which accepts a hyperparameter and an integer and returns that many children of the hyperparameter.
        seed_configurations: an Iterable of the seeds from which we will start beam search.
    
    Returns:
        The best hyperparameter configuration.
    """

    stage = list(seed_configurations)  # TODO: make this a priority queue

    score_of_configuration = dict()

    # for now, we will stop the beam search after we try a set number of configurations.
    # later on, it might be better to stop the beam search when running more stages does not improve the best configuration found.
    n_configurations_to_try = 10
    width = 10  # after every stage, cut down to only this many of the best
    number_of_stages_to_consider = n_configurations_to_try // width 

    # considering the time it takes to train a model, this psuedolinear algorithm doesn't slow things down much.
    # we might want to replace it with a linear-time algorithm, but that isn't very important
    top = lambda iterator, k: sorted(iterator)[:k]

    with Pool(cpu_count()) as pool:
        for _ in range(number_of_stages_to_consider):
            configurations_with_score = pool.map(criterion, stage)

            for configuration, score in configurations_with_score:
                score_of_configuration[configuration] = score

            best_configurations = top(configurations_with_score, width)

            stage = []
            for configuration, score in best_configurations:
                for child in children(configuration, 3):
                    if child not in score_of_configuration:
                        stage.append(child)
    
    best_configuration = max(score_of_configuration, key=score_of_configuration.get)
    return best_configuration 