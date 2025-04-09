import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from Evaluation import evaluation


class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_initializer='glorot_uniform', **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule, self.input_dim_capsule, self.dim_capsule],
            initializer=self.kernel_initializer,
            name='W')

        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = tf.expand_dims(inputs, 2)

        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsule, 1])

        inputs_hat = tf.scan(lambda ac, x: tf.matmul(x, self.W, transpose_b=True), elems=inputs_tiled,
                             initializer=tf.zeros((self.input_num_capsule, self.num_capsule, self.dim_capsule)))

        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.input_num_capsule, self.num_capsule])

        assert self.routings > 0
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=-1)

            outputs = tf.keras.activations.relu(tf.keras.backend.batch_dot(c, inputs_hat, [2, 2]))

            if i < self.routings - 1:
                b += tf.keras.backend.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def Model_CapsNet(train_data, train_target, test_data, test_target, routings=3):
    x = layers.Input(shape=train_data)

    # Layer 1: Conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Convolutional Capsule Layer
    primary_capsule = CapsuleLayer(num_capsule=8, dim_capsule=16, routings=routings, name='primarycaps')(conv1)

    # Layer 3: Capsule Layer
    digit_capsule = CapsuleLayer(num_capsule=train_target.shape[1], dim_capsule=16, routings=routings, name='digitcaps')(
        primary_capsule)

    # Layer 4: Flatten Capsule Layer
    out_capsule = layers.Flatten()(digit_capsule)

    # Decoder network
    y = layers.Input(shape=(train_target.shape[1],))
    masked_by_y = layers.Multiply()([digit_capsule, y])
    masked = layers.Flatten()(masked_by_y)
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * train_target.shape[1]))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(train_data), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=train_target.shape[1], name='out_recon'))

    # Models for training and evaluation (prediction)
    model = models.Model([x, y], [out_capsule, decoder(masked)])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=[tf.keras.losses.sparse_categorical_crossentropy, 'mse'],
                  loss_weights=[1., 0.0005],
                  metrics={'out_caps': 'accuracy'})

    pred = model.predict(test_data)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred





