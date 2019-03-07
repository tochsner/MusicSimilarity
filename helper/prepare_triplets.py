import numpy as np
import random
from data.playlists import *

from .losses_similarity import *

"""
Generates input and output pairs for performing similarity learning with Keras, based on quadruplet-selection.
The grouped data are Spotify-uri's of songs grouped by playlist.
Output format of the Keras model: Embedding ; Output (Flatten), Target Decoder Output
Format of y_train: Target Embedding ; Dissimilar Embedding
"""
def create_quadruplets_for_similarity_learning(model, grouped_data, num_samples, embedding_lenght, slice_width):
    losses = Losses()

    num_classes = len(grouped_data)

    indexes = list(range(num_classes))

    demo_spectrogram = load_all_slices_of_spectrogram("7zZUB3zucFzCMHVAsX7d0Z", slice_width)

    height = demo_spectrogram[0].shape[0]
    width = demo_spectrogram[0].shape[1]

    x_shape = (num_samples, height, width, 1)
    y_shape = (num_samples, 2 * embedding_lenght)

    x_data = np.zeros(x_shape)
    y_data = np.zeros(y_shape)

    for sample in range(num_samples // 2):
        main_index = random.choice(indexes)
        second_index = random.choice([index for index in indexes if index != main_index])
        
        try:
            main_sample1 = load_random_slice_of_spectrogram(random.choice(grouped_data[main_index]), slice_width)
            main_sample2 = load_random_slice_of_spectrogram(random.choice(grouped_data[main_index]), slice_width)
            second_sample1 = load_random_slice_of_spectrogram(random.choice(grouped_data[second_index]), slice_width)
            second_sample2 = load_random_slice_of_spectrogram(random.choice(grouped_data[second_index]), slice_width)
        except:
            continue

        outputs = model.predict(np.array([main_sample1, main_sample2, second_sample1, second_sample2]))

        costs = (losses.get_distance(outputs[0][ : embedding_lenght], outputs[2][ : embedding_lenght]),
                 losses.get_distance(outputs[0][ : embedding_lenght], outputs[3][ : embedding_lenght]),
                 losses.get_distance(outputs[1][ : embedding_lenght], outputs[2][ : embedding_lenght]),
                 losses.get_distance(outputs[1][ : embedding_lenght], outputs[3][ : embedding_lenght]))

        argmin = np.argmin(costs)

        if argmin == 0:
            # mainSample 1
            x_data[2 * sample] = main_sample1    
            y_data[2 * sample, : embedding_lenght] = outputs[1][ : embedding_lenght]
            y_data[2 * sample, embedding_lenght :] = outputs[2][ : embedding_lenght]

            # secondSample 1
            x_data[2 * sample + 1] = second_sample1           
            y_data[2 * sample + 1, : embedding_lenght] = outputs[3][ : embedding_lenght]
            y_data[2 * sample + 1, embedding_lenght :] = outputs[0][ : embedding_lenght]
        elif argmin == 1:
            # mainSample 1
            x_data[2 * sample] = main_sample1    
            y_data[2 * sample, : embedding_lenght] = outputs[1][ : embedding_lenght]
            y_data[2 * sample, embedding_lenght :] = outputs[3][ : embedding_lenght]

            # secondSample 2
            x_data[2 * sample + 1] = second_sample2            
            y_data[2 * sample + 1, : embedding_lenght] = outputs[2][ : embedding_lenght]
            y_data[2 * sample + 1, embedding_lenght :] = outputs[0][ : embedding_lenght]
        elif argmin == 2:
            # mainSample 2
            x_data[2 * sample] = main_sample2   
            y_data[2 * sample, : embedding_lenght] = outputs[0][ : embedding_lenght]
            y_data[2 * sample, embedding_lenght :] = outputs[2][ : embedding_lenght]

            # secondSample 1
            x_data[2 * sample + 1] = second_sample1
            y_data[2 * sample + 1, : embedding_lenght] = outputs[3][ : embedding_lenght]
            y_data[2 * sample + 1, embedding_lenght :] = outputs[1][ : embedding_lenght]
        elif argmin == 3:
            # mainSample 2
            x_data[2 * sample] = main_sample2   
            y_data[2 * sample, : embedding_lenght] = outputs[0][ : embedding_lenght]
            y_data[2 * sample, embedding_lenght :] = outputs[3][ : embedding_lenght]

            # secondSample 2
            x_data[2 * sample + 1] = second_sample2
            y_data[2 * sample + 1, : embedding_lenght] = outputs[2][ : embedding_lenght]
            y_data[2 * sample + 1, embedding_lenght :] = outputs[1][ : embedding_lenght]

    return x_data, y_data


"""
Returns the embedding of an output.
Output format of the Keras model: Embedding ; Output (Flatten) ; Target Decoder Output
"""
def get_embedding(output, embedding_lenght):
    return output[:embedding_lenght]
"""
Returns the decoder output.
Output format of the Keras model: Embedding ; Output (Flatten) ; Target Decoder Output
"""
def get_decoder_output(output, embedding_lenght, decoder_output_lenght):
    return output[embedding_lenght: embedding_lenght + decoder_output_lenght]
"""
Returns the target decoder output.
Output format of the Keras model: Embedding ; Output (Flatten) ; Target Decoder Output
"""
def get_decoder_output(output, embedding_lenght, decoder_output_lenght):
    return output[embedding_lenght + decoder_output_lenght:]


"""
Splits a list into two sublists with ratio r.
"""
def split_list(original_list, r):
    number_samples_total = len(original_list)
    number_samples_1 = int(number_samples_total * r)

    return original_list[:number_samples_1], original_list[number_samples_1:]
