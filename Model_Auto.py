import keras
import numpy as np
from keras import layers
from keras import backend as K


def Model_AutoEn_Feat(data, Target):
    # This is the size of our encoded representations
    encoding_dim = 200  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    # This is our input image
    input_img = keras.Input(shape=(data.shape[1]))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(Target.shape[1], activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    inp = autoencoder.input  # input placeholder
    outputs = [layer.output for layer in autoencoder.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    test = data[:][np.newaxis, ...]
    tets = test[0, :, :, :]
    Feat = np.asarray(functors[1]([tets])).squeeze()
    return Feat