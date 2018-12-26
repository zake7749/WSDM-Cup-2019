from __future__ import absolute_import, division

import tensorflow as tf
from keras.layers import Dense, Input, Embedding, Lambda, Dropout, Activation, SpatialDropout1D, Reshape, GlobalAveragePooling1D, merge, Flatten, Bidirectional, CuDNNGRU, add, Conv1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)


def get_av_cnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)

    filter_nums = 300  # 500->375, 400->373, 300->

    comment_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(comment_input)
    embedded_sequences = SpatialDropout1D(0.25)(embedded_sequences)

    conv_0 = Conv1D(filter_nums, 1, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_1 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_2 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_3 = Conv1D(filter_nums, 4, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)

    attn_0 = AttentionWeightedAverage()(conv_0)
    avg_0 = GlobalAveragePooling1D()(conv_0)
    maxpool_0 = GlobalMaxPooling1D()(conv_0)

    maxpool_1 = GlobalMaxPooling1D()(conv_1)
    attn_1 = AttentionWeightedAverage()(conv_1)
    avg_1 = GlobalAveragePooling1D()(conv_1)

    maxpool_2 = GlobalMaxPooling1D()(conv_2)
    attn_2 = AttentionWeightedAverage()(conv_2)
    avg_2 = GlobalAveragePooling1D()(conv_2)

    maxpool_3 = GlobalMaxPooling1D()(conv_3)
    attn_3 = AttentionWeightedAverage()(conv_3)
    avg_3 = GlobalAveragePooling1D()(conv_3)

    v0_col = merge([maxpool_0, maxpool_1, maxpool_2, maxpool_3], mode='concat', concat_axis=1)
    v1_col = merge([attn_0, attn_1, attn_2, attn_3], mode='concat', concat_axis=1)
    v2_col = merge([avg_1, avg_2, avg_0, avg_3], mode='concat', concat_axis=1)
    merged_tensor = merge([v0_col, v1_col, v2_col], mode='concat', concat_axis=1)
    output = Dropout(0.7)(merged_tensor)
    output = Dense(units=144)(output)
    output = Activation('relu')(output)
    output = Dense(units=out_size, activation='sigmoid')(output)

    model = Model(inputs=comment_input, outputs=output)
    adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model


def get_kmax_text_cnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)

    filter_nums = 180
    drop = 0.6

    comment_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(comment_input)
    embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)

    conv_0 = Conv1D(filter_nums, 1, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_1 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_2 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_3 = Conv1D(filter_nums, 4, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)

    # conv_0 = Conv1D(filter_nums / 2, 1, kernel_initializer="normal", padding="valid", activation="relu")(conv_0)
    # conv_1 = Conv1D(filter_nums / 2, 2, kernel_initializer="normal", padding="valid", activation="relu")(conv_1)
    # conv_2 = Conv1D(filter_nums / 2, 3, strides=2, kernel_initializer="normal", padding="valid", activation="relu")(conv_2)

    maxpool_0 = KMaxPooling(k=3)(conv_0)
    maxpool_1 = KMaxPooling(k=3)(conv_1)
    maxpool_2 = KMaxPooling(k=3)(conv_2)
    maxpool_3 = KMaxPooling(k=3)(conv_3)

    merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2, maxpool_3], mode='concat', concat_axis=1)
    output = Dropout(drop)(merged_tensor)
    output = Dense(units=144, activation='relu')(output)
    output = Dense(units=out_size, activation='sigmoid')(output)

    model = Model(inputs=comment_input, outputs=output)
    adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model


def get_rcnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    recurrent_units = 64
    filter1_nums = 128

    input_layer = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)(input_layer)

    embedding_layer = SpatialDropout1D(0.2)(embedding_layer)
    rnn_1 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)

    #conv_1 = Conv1D(filter1_nums, 1, kernel_initializer="uniform", padding="valid", activation="relu", strides=1)(rnn_1)
    #maxpool = GlobalMaxPooling1D()(conv_1)
    #attn = AttentionWeightedAverage()(conv_1)
    #average = GlobalAveragePooling1D()(conv_1)

    conv_2 = Conv1D(filter1_nums, 2, kernel_initializer="normal", padding="valid", activation="relu", strides=1)(rnn_1)
    maxpool = GlobalMaxPooling1D()(conv_2)
    attn = AttentionWeightedAverage()(conv_2)
    average = GlobalAveragePooling1D()(conv_2)

    concatenated = concatenate([maxpool, attn, average], axis=1)
    x = Dropout(0.5)(concatenated)
    x = Dense(120, activation="relu")(x)
    output_layer = Dense(out_size, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=5, decay=1e-5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model


def get_siamese_av_rnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    recurrent_units = 60

    # -- Dual input part -- 
    print("Prepareing Inputs")
    input_layer_1 = Input(shape=(max_sequence_length,), name='first_sentences')
    input_layer_2 = Input(shape=(max_sequence_length,), name='second_sentences')
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)
    
    # -- Denstify -- 
    embedded_1 = embedding_layer(input_layer_1)
    embedded_1 = SpatialDropout1D(0.25)(embedded_1)

    embedded_2 = embedding_layer(input_layer_2)
    embedded_2 = SpatialDropout1D(0.25)(embedded_2)

    # -- prepare network --
    rnn_1 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))
    rnn_2 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))
    last = Lambda(lambda t: t[:, -1], name='last')
    max_pool = GlobalMaxPooling1D()
    attn_pool = AttentionWeightedAverage()
    avg_pool = GlobalAveragePooling1D()

    ffn = Dense(72, activation='relu')

    sent_vectors = []
    for embedded in [embedded_1, embedded_2]:
        print("HUCK")
        rnn_1_out = rnn_1(embedded)
        rnn_2_out = rnn_2(rnn_1_out)
        rnn_out = concatenate([rnn_1_out, rnn_2_out], axis=2)

        last_vec = last(rnn_out)
        max_vec = max_pool(rnn_out)
        attn_vec = attn_pool(rnn_out)
        avg_vec = avg_pool(rnn_out)

        all_views = concatenate([last_vec, max_vec, attn_vec, avg_vec], axis=1)
        x = Dropout(0.5)(all_views)
        x = ffn(x)
        sent_vectors.append(x)

    sent_vectors_pairs = concatenate(sent_vectors, axis=1)
    x = Dense(72, activation="relu")(sent_vectors_pairs)
    output_layer = Dense(out_size, activation="sigmoid")(x)

    print("DONE")
    model = Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer)
    adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model