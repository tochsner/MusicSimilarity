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

spectrogram_height = 263

"""
Imports a list of the songs in each playlist.
"""
def load_playlists():
    with open(playlists_path, "r") as f:
        return [np.array(x.strip().split(','))[2:] for x in f.readlines()]


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


"""
Returns an image augmentator for spectrograms.
"""
def get_image_data_generator():
    return ImageDataGenerator(
            height_shift_range=50)
