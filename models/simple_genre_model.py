from keras import backend as k
from keras.models import Input, Model
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Dropout, MaxPooling2D, Lambda, Concatenate, BatchNormalization

"""
Builds a simple convnet for music information retreival out of a spectrogram.
"""
def std_layer(x):
    return k.var(x, axis=2, keepdims=True)
def min_layer(x):
    return k.min(x, axis=2, keepdims=True)

def build_model(input_shape, output_lenght, params):
    height = input_shape[0]
    width = input_shape[1]


def build_model(input_shape, output_lenght):
    height = input_shape[0]
    width = input_shape[1]

    inputLayer = Input(input_shape)
    convLayer1 = Conv2D(50, (height, 1), activation='relu', name='conv1')(inputLayer)
    convLayer1 = BatchNormalization()(convLayer1)
    convLayer2 = Conv2D(50, (1, 4), activation='relu', name='conv2')(convLayer1)
    convLayer2 = BatchNormalization()(convLayer2)
    convLayer3 = Conv2D(100, (1, 4), activation='relu', name='conv3')(convLayer2)
    convLayer3 = BatchNormalization()(convLayer3)
        
    avgLayer = AveragePooling2D((1, width - 6))(convLayer3)
    maxLayer = MaxPooling2D((1, width - 6))(convLayer3)
    minLayer = Lambda(min_layer)(convLayer3)
    stdLayer = Lambda(std_layer)(convLayer3)
    concatenated = Concatenate()([avgLayer, minLayer, maxLayer, stdLayer])
    flatten = Flatten()(concatenated)
    dense = Dense(150, activation='relu', name='dense1')(flatten)
    dense = BatchNormalization()(dense)
    dense = Dense(output_lenght, activation='softmax', name='dense2')(dense)
    model = Model(inputs=inputLayer, outputs=dense)
    
    return model
