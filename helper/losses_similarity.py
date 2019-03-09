"""
The custom losses for similarity with siamese networks with keras.
Output format of the Keras model (y_pred): Embedding ; Decoder Output (Flatten)
Format of y_true: Target Embedding ; Dissimlilar Embedding ; Target Decoder Output
"""

from keras import backend as K
from .prepare_triplets import *


class Losses():
    def __init__(self, embedding_lenght=0, decoder_output_length=0, decoder_factor=0.5):
        self.mse = MeanSquareCostFunction()

        self.embedding_length = embedding_lenght
        self.decoder_output_length = decoder_output_length

        self.decoder_factor = decoder_factor
        self.trio_factor = 1 - self.decoder_factor

    def trio_loss(self, y_true, y_pred):
        output_embedding = get_embedding(y_pred, self.embedding_length)
        target_embedding = get_embedding(y_true, self.embedding_length)
        dissimilar_embedding = get_dissimilar_embedding(y_true, self.embedding_length)

        return K.sum(K.square(output_embedding - target_embedding), axis=-1) - \
               K.sum(K.square(output_embedding - dissimilar_embedding), axis=-1)
    
    def quadruplet_loss(self, y_true, y_pred):
        output_embedding = get_embedding(y_pred, self.embedding_length)
        target_embedding = get_embedding(y_true, self.embedding_length)
        dissimilar_embedding = get_dissimilar_embedding(y_true, self.embedding_length)

        decoder_output = get_decoder_output(y_pred, self.embedding_length, self.decoder_output_length)
        target_decoder_output = get_similar_decoder_output(y_true, self.embedding_length)
        
        return self.decoder_factor * K.sum(K.square(decoder_output - target_decoder_output), axis=-1) + \
               self.trio_factor * K.sum(K.square(output_embedding - target_embedding), axis=-1) + \
               self.trio_factor * K.sum(K.square(output_embedding) / 2 - output_embedding * dissimilar_embedding -
                                        K.abs(output_embedding - dissimilar_embedding), axis=-1)
    
    def quadruplet_metric(self, y_true, y_pred):
        output_embedding = get_embedding(y_pred, self.embedding_length)
        target_embedding = get_embedding(y_true, self.embedding_length)
        dissimilar_embedding = get_dissimilar_embedding(y_true, self.embedding_length)
        
        return K.mean(K.cast(K.less(K.sum(K.square(output_embedding - target_embedding), axis=-1),
                                    K.sum(K.square(output_embedding - dissimilar_embedding), axis=-1)), 'float16'))
