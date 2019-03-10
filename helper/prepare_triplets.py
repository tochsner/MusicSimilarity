from data.playlists import *

from .losses import *

"""
Generates input and output pairs for performing similarity learning with Keras, based on quadruplet-selection.
The grouped data are Spotify-uri's of songs grouped by playlist.
Output format of the Keras model: Embedding ; Decoder Output ; Target Decoder Output
Format of y_true: Similar Embedding ; Dissimilar Embedding ; Similar Decoder Output
"""
def create_quadruplets_for_similarity_learning(model, grouped_data, num_samples, embedding_length,
                                               decoder_output_length, slice_width):
    mse = MeanSquareCostFunction()

    num_classes = len(grouped_data)

    indexes = list(range(num_classes))

    demo_spectrogram = load_all_slices_of_spectrogram("7zZUB3zucFzCMHVAsX7d0Z", slice_width)

    height = demo_spectrogram[0].shape[0]
    width = demo_spectrogram[0].shape[1]

    x_shape = (num_samples, height, width, 1)
    y_shape = (num_samples, 2 * embedding_length + decoder_output_length)

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

        main_embedding_1 = get_embedding(outputs[0], embedding_length)
        main_embedding_2 = get_embedding(outputs[1], embedding_length)
        second_embedding_1 = get_embedding(outputs[2], embedding_length)
        second_embedding_2 = get_embedding(outputs[3], embedding_length)

        main_decoder_output_1 = get_target_decoder_output(outputs[0], embedding_length, decoder_output_length)
        main_decoder_output_2 = get_target_decoder_output(outputs[1], embedding_length, decoder_output_length)
        second_decoder_output_1 = get_target_decoder_output(outputs[2], embedding_length, decoder_output_length)
        second_decoder_output_2 = get_target_decoder_output(outputs[3], embedding_length, decoder_output_length)

        costs = (mse.get_cost(main_embedding_1, second_embedding_1),
                 mse.get_cost(main_embedding_1, second_embedding_2),
                 mse.get_cost(main_embedding_2, second_embedding_1),
                 mse.get_cost(main_embedding_2, second_embedding_2))

        arg_min = np.argmin(costs)

        if arg_min == 0:
            # mainSample 1
            x_data[2 * sample] = main_sample1    
            y_data[2 * sample] = get_target_output(main_embedding_2, second_embedding_1, main_decoder_output_2)

            # secondSample 1
            x_data[2 * sample + 1] = second_sample1
            y_data[2 * sample + 1] = get_target_output(second_embedding_2, main_embedding_1, second_decoder_output_2)
        elif arg_min == 1:
            # mainSample 1
            x_data[2 * sample] = main_sample1
            y_data[2 * sample] = get_target_output(main_embedding_2, second_embedding_2, main_decoder_output_2)

            # secondSample 2
            x_data[2 * sample + 1] = second_sample2
            y_data[2 * sample + 1] = get_target_output(second_embedding_1, main_embedding_1, second_decoder_output_1)
        elif arg_min == 2:
            # mainSample 2
            x_data[2 * sample] = main_sample2
            y_data[2 * sample] = get_target_output(main_embedding_1, second_embedding_1, main_decoder_output_1)

            # secondSample 1
            x_data[2 * sample + 1] = second_sample1
            y_data[2 * sample + 1] = get_target_output(second_embedding_2, main_embedding_2, second_decoder_output_2)
        elif arg_min == 3:
            # mainSample 2
            x_data[2 * sample] = main_sample2
            y_data[2 * sample] = get_target_output(main_embedding_1, second_embedding_2, main_decoder_output_1)

            # secondSample 2
            x_data[2 * sample + 1] = second_sample2
            y_data[2 * sample + 1] = get_target_output(second_embedding_1, main_embedding_2, second_decoder_output_1)

    return x_data, y_data


"""
Returns the embedding from a Keras output or y_true.
"""
def get_embedding(output, embedding_length):
    if len(output.shape) == 1:
        return output[:embedding_length]
    else:
        return output[:, :embedding_length]


"""
Returns the dissimilar embedding from y_true.
"""
def get_dissimilar_embedding(y_true, embedding_length):
    if len(y_true.shape) == 1:
        return y_true[embedding_length:2*embedding_length]
    else:
        return y_true[:, embedding_length:2 * embedding_length]


"""
Returns the decoder output from a Keras output.
"""
def get_decoder_output(output, embedding_length, decoder_output_length):
    if len(output.shape) == 1:
        return output[embedding_length:embedding_length + decoder_output_length]
    else:
        return output[:, embedding_length:embedding_length + decoder_output_length]


"""
Returns the target decoder output from a Keras output.
"""
def get_target_decoder_output(output, embedding_length, decoder_output_length):
    if len(output.shape) == 1:
        return output[embedding_length + decoder_output_length:]
    else:
        return output[:, embedding_length + decoder_output_length:]


"""
Returns the decoder output of a similar song from y_true.
"""
def get_similar_decoder_output(y_true, embedding_length):
    if len(y_true.shape) == 1:
        return y_true[2*embedding_length:]
    else:
        return y_true[:, 2*embedding_length:]


"""
Returns a target output (y_true).
Unlike the Target Decoder Output of the Keras model, the Target Decoder Output of y_true
can be the one of a similar song, used for a cross-song encoder.
"""
def get_target_output(similar_embedding, dissimilar_embedding, target_decoder_output):
    return np.concatenate([similar_embedding, dissimilar_embedding, target_decoder_output])
