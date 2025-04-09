import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np
from keras import backend as K
from sklearn.naive_bayes import GaussianNB
from Evaluation import evaluation


# Capsule Layer Class
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = tf.random_normal_initializer(mean=0., stddev=1.)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        self.W = self.add_weight(shape=[self.num_capsules, self.input_num_capsules,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

    def call(self, inputs):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsules, 1, 1])
        inputs_tiled = tf.expand_dims(inputs_tiled, 4)

        inputs_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled)

        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsules, self.input_num_capsules])

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            c = tf.expand_dims(c, -1)
            c = tf.expand_dims(c, 2)

            s = tf.multiply(inputs_hat, c)
            s = tf.reduce_sum(s, axis=2, keepdims=True)
            v = squash(s)

            b += tf.matmul(inputs_hat, tf.transpose(v, perm=[0, 1, 3, 2]))

        return tf.squeeze(v, axis=[2])


def squash(x, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(x), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return scale * x


# Define the CapsNet Model
def Model_capsnet_Feat(image, Tar, routings=3):
    inputs = layers.Input(shape=image)
    x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Reshape((-1, 128))(x)
    capsule = CapsuleLayer(num_capsules=Tar.shape[0], dim_capsule=16, routings=routings)(x)
    outputs = layers.Lambda(lambda z: tf.sqrt(tf.reduce_sum(tf.square(z), axis=2)))(capsule)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    test = image[:][np.newaxis, ...]
    tets = test[0, :, :, :]
    layer_out = np.asarray(functors[0]([tets])).squeeze()
    return layer_out


def Model_HDMcCNe(Data, Target):
    Feature = Model_capsnet_Feat(Data, Target)

    learnperc = round(Feature.shape[0] * 0.75)  # Split Training and Testing Datas
    train_data = Feature[:learnperc, :]
    train_target = Target[:learnperc, :]
    test_data = Feature[learnperc:, :]
    test_target = Target[learnperc:, :]

    # Initialize the Gaussian Naive Bayes classifier
    classifier = GaussianNB()
    # Train the classifier
    classifier.fit(train_data, train_target)
    # Predict on the test set
    pred = classifier.predict(test_target)
    Eval = evaluation(pred, test_target)
    return Eval, pred


