from keras import backend as k
from keras.models import Input, Model
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D,\
    MaxPooling2D, Lambda, Concatenate, BatchNormalization, Activation

"""
Builds a simple convnet for music similarity with a quadruplet cross-playlist encoder.
"""


def std_layer_function(x):
    return k.var(x, axis=2, keepdims=True)


def sigmoid_positive(x):
    return 2 * (k.sigmoid(x) - 0.5)


def build_model(input_shape, embedding_length, decoder_output_length):
    height = input_shape[0]
    width = input_shape[1]

    input_layer = Input(input_shape)
    conv_layer1 = Conv2D(60, (height, 1), activation='relu', name='conv1', trainable=True)(input_layer)
    conv_layer1 = BatchNormalization()(conv_layer1)
    conv_layer2 = Conv2D(60, (1, 4), activation='relu', name='conv2', trainable=True)(conv_layer1)
    conv_layer2 = BatchNormalization()(conv_layer2)
    conv_layer3 = Conv2D(60, (1, 4), activation='relu', name='conv3', trainable=True)(conv_layer2)
    conv_layer3 = BatchNormalization()(conv_layer3)

    avg_layer1 = AveragePooling2D((1, width))(conv_layer1)
    avg_layer2 = AveragePooling2D((1, width - 3))(conv_layer2)
    avg_layer3 = AveragePooling2D((1, width - 6))(conv_layer3)
    max_layer3 = MaxPooling2D((1, width - 6))(conv_layer3)
    std_layer3 = Lambda(std_layer_function)(conv_layer3)

    concatenated_avg = Concatenate()([avg_layer1, avg_layer2, avg_layer3])
    concatenated3 = Concatenate()([avg_layer3, std_layer3, max_layer3])

    flatten_avg = Flatten()(concatenated_avg)
    flatten3 = Flatten()(concatenated3)

    flatten_avg = Lambda(sigmoid_positive)(flatten_avg)

    dense = Dense(120, activation='relu')(flatten3)
    encoder_output = Dense(embedding_length, activation='sigmoid')(dense)

    dense = Dense(120, activation='relu')(encoder_output)
    decoder_output = Dense(decoder_output_length, activation='sigmoid')(dense)

    output_layer = Concatenate()([encoder_output, decoder_output, flatten_avg])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
