"""
Allows the client to run a hyperparameter search with keras.
"""

import itertools
<<<<<<< HEAD
import random
from .logger import Logger
=======
from logger.py import Logger
>>>>>>> 901c6d9fcb21b6d1858221b7de6e371e76989e23

class HyperparameterSearch:
    def __init__(self, name):
        self.logger = Logger(name)

    """
<<<<<<< HEAD
    Runs the search. Tests a certain percentage of all possible permutations of the parameters.
=======
    Runs the search. Tests a certain percentage of all possible permuattaions of the parameters.
>>>>>>> 901c6d9fcb21b6d1858221b7de6e371e76989e23
    training_function(parameter) has to be a function which trains a model with keras and returns a history object.
    parameter has to be a dictionary with possible parameters.
    """
    def scan(self, training_function, parameter, percentage_tested):
        parameter_permutations = self.get_permutations(parameter, percentage_tested)
        number_rounds = len(parameter_permutations)

<<<<<<< HEAD
        print("Start 0 / " + str(number_rounds))

        for r in range(number_rounds):
            print(parameter_permutations[r])
            print(str(r) + " / " + str(number_rounds))

=======
        for r in range(rounds):
>>>>>>> 901c6d9fcb21b6d1858221b7de6e371e76989e23
            result = training_function(parameter_permutations[r])
            self.log_results(parameter_permutations[r], result)

    def get_permutations(self, parameter, percentage):
        keys, values = zip(*parameter.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
<<<<<<< HEAD
        number_permutations = len(permutations)

        random.shuffle(permutations)
        permutations = permutations[ : int(number_permutations * percentage)]
=======
>>>>>>> 901c6d9fcb21b6d1858221b7de6e371e76989e23

        return permutations

    def log_results(self, parameter, result):
<<<<<<< HEAD
        self.logger.log(str(parameter) + " " + str(result))
=======
        self.logger.log(str(parameter) + " " + )
>>>>>>> 901c6d9fcb21b6d1858221b7de6e371e76989e23
