import random
import numpy as np

"""
Splits the list into two random sublists with ratio r.
"""
def split_list(original_list, r):
    random.shuffle(original_list)

    number_samples_total = len(original_list)
    number_samples_1 = int(number_samples_total * r)

    return original_list[:number_samples_1], original_list[number_samples_1:]


class OutputHelper:

    def __init__(self, embedding_length, decoder_output_length):
        self.embedding_length = embedding_length
        self.decoder_output_length = decoder_output_length


    """
    Returns the embedding from a Keras output or y_true.
    """
    def get_embedding(self, output):
        if len(output.shape) == 1:
            return output[: self.embedding_length]
        else:
            return output[:, : self.embedding_length]


    """
    Returns the dissimilar embedding from y_true.
    """
    def get_dissimilar_embedding(self, y_true):
        if len(y_true.shape) == 1:
            return y_true[self.embedding_length : 2 * self.embedding_length]
        else:
            return y_true[:, self.embedding_length : 2 * self.embedding_length]


    """
    Returns the decoder output from a Keras output.
    """
    def get_decoder_output(self, output):
        if len(output.shape) == 1:
            return output[self.embedding_length : self.embedding_length + self.decoder_output_length]
        else:
            return output[:, self.embedding_length : self.embedding_length + self.decoder_output_length]


    """
    Returns the target decoder output from a Keras output.
    """
    def get_target_decoder_output(self, output):
        if len(output.shape) == 1:
            return output[self.embedding_length + self.decoder_output_length:]
        else:
            return output[:, self.embedding_length + self.decoder_output_length:]


    """
    Returns the decoder output of a similar song from y_true.
    """
    def get_similar_decoder_output(self, y_true):
        if len(y_true.shape) == 1:
            return y_true[2 * self.embedding_length:]
        else:
            return y_true[:, 2 * self.embedding_length:]


    """
    Returns a target output (y_true).
    Unlike the Target Decoder Output of the Keras model, the Target Decoder Output of y_true
    can be the one of a similar song, used for a cross-song encoder.
    """
    def get_target_output(self, similar_embedding, dissimilar_embedding, target_decoder_output):
        return np.concatenate([similar_embedding, dissimilar_embedding, target_decoder_output])
