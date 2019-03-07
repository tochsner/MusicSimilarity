"""
Loads the 2821-playlists data set based on Spotify playlists containing over 100'000 songs.
Data gets formatted for the use with Keras and few-shot learning.
"""

import random
import numpy as np
from skimage.io import imread, imsave
from keras.preprocessing.image import ImageDataGenerator

spectrogram_path = "/home/tobia/Documents/ML/Data MA Sono"
spectrogram_type = ".png"
playlists_path = "data/Playlists.csv"

spectrogram_height = 250

"""
Imports a list of the songs in each playlist.
"""
def load_playlists():
    with open(playlists_path, "r") as f:
        return [np.array(x.strip().split(','))[1:] for x in f.readlines()]


"""
Loads the spectrogram for a specific Spotify URI (id).
"""
def load_spectrogram(uri):
    spectrogram = imread(spectrogram_path + "/" + str(uri) + spectrogram_type) / 256

    height = spectrogram.shape[0]
    width = spectrogram.shape[1]

    spectrogram = spectrogram[height - spectrogram_height:]

    return spectrogram.reshape(spectrogram_height, width, 1)


def load_random_slice_of_spectrogram(uri, slice_width):
    spectrogram = load_spectrogram(uri)

    width = spectrogram.shape[1]
    start = int(random.random() * (width - slice_width))

    return spectrogram[:, start : start + slice_width]


def load_all_slices_of_spectrogram(uri, slice_width):    
    spectrogram = load_spectrogram(uri)

    width = spectrogram.shape[1]    

    return [spectrogram[:, start_index : start_index + slice_width] for start_index in range(0, width - slice_width + 1, slice_width)]


def load_data_for_keras(slice_width, ratio = 0.7, percentage_of_spectrograms_used = 1):
    genres = load_playlists()

    genres_count = len(genres)
    songs_count = sum([len(g) for g in genres])

    demo_spectrogram = load_all_slices_of_spectrogram(genres[0][0], slice_width)

    height = demo_spectrogram[0].shape[0]
    width = demo_spectrogram[0].shape[1]

    training_slices_count = int(songs_count * len(demo_spectrogram) * ratio * percentage_of_spectrograms_used)
    test_slices_count = int(songs_count * len(demo_spectrogram) * (1 - ratio) * percentage_of_spectrograms_used)

    #uses channels_last
    x_train = np.zeros((training_slices_count, height, width, 1))
    y_train = np.zeros((training_slices_count, genres_count))

    x_test = np.zeros((test_slices_count, height, width, 1))
    y_test = np.zeros((test_slices_count, genres_count))

    counter_train = 0
    counter_test = 0

    for g in range(genres_count):
        for song in genres[g]:
            try:
                all_slices = load_all_slices_of_spectrogram(song, slice_width)

                if random.random() < ratio:
                    for slice in all_slices:
                        if random.random() < percentage_of_spectrograms_used and counter_train < training_slices_count:
                            x_train[counter_train] = slice
                            y_train[counter_train] = np.zeros((genres_count))
                            y_train[counter_train, g] = 1
                            counter_train += 1
                else:
                    for slice in all_slices:
                        if random.random() < percentage_of_spectrograms_used and counter_test < test_slices_count:
                            x_test[counter_test] = slice
                            y_test[counter_test] = np.zeros((genres_count))
                            y_test[counter_test, g] = 1
                            counter_test += 1
            except:
                continue

    x_train = x_train[:counter_train]
    y_train = y_train[:counter_train]
    x_test = x_test[:counter_test]
    y_test = y_test[:counter_test]

    return ((x_train, y_train), (x_test, y_test))


"""
Returns an image augmentator for spectrograms.
"""
def get_image_data_generator():
    return ImageDataGenerator(
            height_shift_range=50)
