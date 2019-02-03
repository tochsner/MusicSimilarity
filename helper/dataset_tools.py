import numpy as np
import random

"""
Splits the dataset (data=(x_data, y_data)) into two parts of ratio % / 100% - ratio %.
"""
def split_dataset(data, ratio, seed = -1):
    if seed == -1:
        seed = random.randint(0, 1000)

    (x_data, y_data) = data

    number_of_samples = data[0].shape[0]
    number_of_samples_1 = int(number_of_samples * ratio)

    np.random.seed(seed)
    np.random.shuffle(x_data)
    np.random.seed(seed)
    np.random.shuffle(y_data)

    x_data_1 = x_data[ : number_of_samples_1]
    y_data_1 = y_data[ : number_of_samples_1]
    x_data_2 = x_data[number_of_samples_1 : ]
    y_data_2 = y_data[number_of_samples_1 : ]

    return ((x_data_1, y_data_1), (x_data_2, y_data_2))