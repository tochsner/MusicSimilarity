import random

"""
Splits the list into two random sublists with ratio r.
"""
def split_list(original_list, r):
    random.shuffle(original_list)

    number_samples_total = len(original_list)
    number_samples_1 = int(number_samples_total * r)

    return original_list[:number_samples_1], original_list[number_samples_1:]
