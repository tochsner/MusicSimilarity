"""
The custom losses for similarity with siamese networks with keras.
"""

from keras import backend as K
from .prepare_triplets import *

class Losses:
    def __init__(self, output_helper, decoder_factor=0.5):
        self.output_helper = output_helper

        self.decoder_factor = decoder_factor
        self.trio_factor = 1 - self.decoder_factor


    def trio_loss(self, y_true, y_pred):
        output_embedding = self.output_helper.get_embedding(y_pred)
        target_embedding = self.output_helper.get_embedding(y_true)
        dissimilar_embedding = self.output_helper.get_dissimilar_embedding(y_true)

        return K.mean(K.square(output_embedding - target_embedding), axis=-1) + \
               K.mean(K.square(output_embedding) / 2 - output_embedding * dissimilar_embedding - \
                                        K.abs(output_embedding - dissimilar_embedding), axis=-1)


    def quadruplet_loss(self, y_true, y_pred):
        output_embedding = self.output_helper.get_embedding(y_pred)
        target_embedding = self.output_helper.get_embedding(y_true)
        dissimilar_embedding = self.output_helper.get_dissimilar_embedding(y_true)

        decoder_output = self.output_helper.get_decoder_output(y_pred)
        target_decoder_output = self.output_helper.get_similar_decoder_output(y_true)

        return self.decoder_factor * K.mean(K.square(decoder_output - target_decoder_output), axis=-1) + \
               self.trio_factor * K.mean(K.square(output_embedding - target_embedding), axis=-1) + \
               self.trio_factor * K.mean(K.square(output_embedding) / 2 - output_embedding * dissimilar_embedding -
                                        K.abs(output_embedding - dissimilar_embedding), axis=-1)


    def quadruplet_metric(self, y_true, y_pred):
        output_embedding = self.output_helper.get_embedding(y_pred)
        target_embedding = self.output_helper.get_embedding(y_true)
        dissimilar_embedding = self.output_helper.get_dissimilar_embedding(y_true)

        return K.mean(K.cast(K.less(K.sum(K.square(output_embedding - target_embedding), axis=-1),
                                    K.sum(K.square(output_embedding - dissimilar_embedding), axis=-1)), K.floatx()))
