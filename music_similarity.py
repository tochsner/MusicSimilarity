"""
Trains a quadruplet cross-playlist encoder on Spotify playlist data.
This model can be used with or without pre-training.
"""

from models.simple_genre_model import *
from helper.prepare_triplets import *
from keras.optimizers import Adam

decoder_factor = 0.6

epochs = 30
batch_size = 32
batches_per_epoch = 100
split_ratio = 0.8
num_test_samples = 2000

slice_width = 40
embedding_length = 20
decoder_output_length = 540

input_shape = (spectrogram_height, slice_width, 1)

losses = Losses(embedding_length, decoder_output_length, decoder_factor)

playlists = load_playlists()
playlists_train, playlists_test = split_list(playlists, split_ratio)

model = build_model(input_shape, embedding_length, decoder_output_length)

model.compile(loss=losses.quadruplet_loss,
              optimizer=Adam(),
              metrics=[losses.quadruplet_metric])

test_data = create_quadruplets_for_similarity_learning(model, playlists_test, num_test_samples,
                                                       embedding_length, slice_width)

for e in range(epochs):
    x_data, y_data = create_quadruplets_for_similarity_learning(model, playlists_train,
                                                                batch_size * batches_per_epoch,
                                                                embedding_length, slice_width)

    model.fit(x_data, y_data, batch_size=batch_size, epochs=1, validation_data=test_data)
