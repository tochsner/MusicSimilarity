"""
Trains a quadruplet cross-playlist encoder on Spotify playlist data.
This model can be used with or without pre-training.
"""

from models.simple_genre_model import *
from helper.dataset_tools import *
from helper.losses_similarity import *
import tensorflow as tf

decoder_factor = 0.6

epochs = 30
batch_size = 16
batches_per_epoch = 100
split_ratio = 0.8
num_test_samples = 1000

slice_width = 40
embedding_length = 20
decoder_output_length = 540

input_shape = (spectrogram_height, slice_width, 1)

losses = Losses(embedding_length, decoder_output_length, decoder_factor)

playlists = load_playlists()
playlists_train, playlists_test = split_list(playlists, split_ratio)

model = build_model(input_shape, embedding_length, decoder_output_length)

model.compile(loss=losses.trio_loss,
              optimizer='sgd',
              metrics=[losses.quadruplet_metric])

model.load_weights("/home/tobia/Documents/ML/Genre-Classification/augmented_final_0", by_name=True)

test_data = create_quadruplets_for_similarity_learning(model, playlists_test, num_test_samples,
                                                       embedding_length, decoder_output_length, slice_width)

predictions = model.predict(test_data[0])

with tf.Session() as sess:
    print(losses.quadruplet_metric(test_data[1], predictions).eval())

def training_sample_generator():
    while True:
        yield create_quadruplets_for_similarity_learning(model, playlists_train, batch_size,
                                                         embedding_length, decoder_output_length, slice_width)


model.fit_generator(training_sample_generator(), epochs, steps_per_epoch=batches_per_epoch, validation_data=test_data)
