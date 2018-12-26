import random, os, sys
import re
import csv
import codecs
import numpy as np
np.random.seed(1337)

import pandas as pd
import operator
import sys

from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import iwillwin.model.model_zoos as model_zoos
from iwillwin.trainer.supervised_trainer import KerasModelTrainer
from iwillwin.data_utils.data_helpers import DataTransformer, DataLoader
from iwillwin.config import dataset_config, model_config

import tensorflow as tf
from keras.layers import Dense, Input, MaxPooling1D, CuDNNLSTM, Embedding, Add, Lambda, Dropout, Activation, SpatialDropout1D, Reshape, GlobalAveragePooling1D, merge, Flatten, Bidirectional, CuDNNGRU, add, Conv1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.legacy.layers import Highway
from keras.layers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam, Adam
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.initializers import *
from keras.activations import softmax
from keras.regularizers import l2
from keras.layers.merge import concatenate

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super().build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []
        for _ in range(n_head):
            self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
            self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
            self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization()
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        n_head = self.n_head
        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)   
            ks = self.ks_layers[i](k) 
            vs = self.vs_layers[i](v) 
            head, attn = self.attention(qs, ks, vs, mask)
            heads.append(head)
            attns.append(attn)
        head = Concatenate()(heads)
        attn = Concatenate()(attns)
        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn

class SelfAttention():
    def __init__(self, d_model=300, d_inner_hid=300, n_head=8, d_k=50, d_v=50, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

    def __call__(self, src_seq, enc_input):
        mask = Lambda(lambda x: get_padding_mask(x, x))(src_seq)
        output, self_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        return output

def get_padding_mask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2,1])
    return mask

def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape

def substract(input_1, input_2):
    "Substract element-wise"
    out_ = Lambda(lambda x: x[0] - x[1])([input_1, input_2])
    return out_

def eldistance(input_1, input_2):
    "Substract element-wise"
    out_ = Lambda(lambda x: K.sqrt(K.square(x[0] - x[1])))([input_1, input_2])
    return out_

def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    add = Add()([input_1, input_2])
    sub = substract(input_1, input_2)
    distance = eldistance(input_1, input_2)
    
    dual = Concatenate()([input_1, input_2])
    dual = Dense(32, activation='relu')(dual)
    dual = Dropout(0.5)(dual)
    dual = Dense(8, activation='relu')(dual)
    
    out_= Concatenate()([sub, mult, add,])
    return out_

def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_

def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_

def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                             output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

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

def get_cnns(seq_length, length, feature_maps, kernels):
    cnns, pools = [], []
    for feature_map, kernel in zip(feature_maps, kernels):
        reduced_l = length - kernel + 1
        conv = Conv2D(feature_map, (1, kernel), activation='relu', data_format="channels_last")
        cnns.append(conv)
        maxp = MaxPooling2D((1, reduced_l), data_format="channels_last")
        pools.append(maxp)
    return cnns, pools

def char_cnn(cnns, pools, length, seq_length, feature_maps, char_embeddings):

    concat_input = []
    for i in range(len(cnns)):
        conved = cnns[i](char_embeddings)
        pooled = pools[i](conved)
        concat_input.append(pooled)

    x = Concatenate()(concat_input)
    x = Reshape((seq_length, sum(feature_maps)))(x)
    x = Dropout(0.1)(x)
    return x

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


def get_decomp_attn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
                           projection_dim=50, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=300, compare_dropout=0.2,
                           dense_dim=256, dense_dropout=0.2,
                           lr=1e-3, activation='relu'):
    
    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q1_c = Input(shape=(max_sequence_length, 11), name='first_sentences_char')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q2_c = Input(shape=(max_sequence_length, 11), name='second_sentences_char')
    input_layer_3 = Input(shape=(len(model_config.META_FEATURES),), name='mata-features', dtype="float32")
    
    embedding = Embedding(nb_words,
                            embedding_dim,
                            input_length=max_sequence_length,
                            trainable=True)
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.5)(q1_embed)
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.5)(q2_embed)
    
    self_attention = SelfAttention()
    
    th = TimeDistributed(Highway(activation='relu'))
    th_2 = TimeDistributed(Highway(activation='relu'))
    
    q1_encoded = th(q1_embed,)
    q2_encoded = th(q2_embed,)
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)    
    
    # Compare Deep
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned),])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned),]) 
    
    compare_layers_d = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    
    q1_compare = time_distributed(q1_combined, compare_layers_d)
    q2_compare = time_distributed(q2_combined, compare_layers_d)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D(),])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D(),])

    # Dense meta featues
    meta_densed = BatchNormalization()(input_layer_3)
    meta_densed = Highway(activation='relu')(meta_densed)
    meta_densed = Dropout(0.2)(meta_densed)
    
    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    q_rep = Concatenate()([q1_rep, q2_rep])
    
    h_all = Concatenate()([q_diff, q_multi, q_rep, meta_densed])
    h_all = Dropout(0.52)(h_all)
    
    dense = Dense(dense_dim, activation=activation)(h_all)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    
    out_ = Dense(3, activation='softmax')(dense)
    
    model = Model(inputs=[q1, q2, input_layer_3], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    model.summary()
    return model

def get_dense_cnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q1_c = Input(shape=(max_sequence_length, 11), name='first_sentences_char')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q2_c = Input(shape=(max_sequence_length, 11), name='second_sentences_char')
    input_layer_3 = Input(shape=(len(model_config.META_FEATURES),), name='mata-features', dtype="float32")

    embedding = Embedding(nb_words,
                            embedding_dim,
                            
                            input_length=max_sequence_length,
                            trainable=True)
    
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.5)(q1_embed)
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.5)(q2_embed)

    th = TimeDistributed(Highway(activation='relu'))
    
    q1_encoded = th(q1_embed,)    
    q2_encoded = th(q2_embed,)
    
    cnn_init = Conv1D(32, 1, strides=1, padding='same', activation='relu')

    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    q1_encoded = Concatenate()([q2_aligned, q1_encoded])
    q2_encoded = Concatenate()([q1_aligned, q2_encoded])     
    norm = BatchNormalization()

    q1_seq = cnn_init(q1_encoded)
    q1_seq = norm(q1_seq)
    q2_seq = cnn_init(q2_encoded)    
    q2_seq = norm(q2_seq)
    
    cnn_2 = Conv1D(32, 3, strides=1, padding='same', activation='relu')
    cnn_3 = Conv1D(32, 1, strides=1, padding='same', activation='relu')

    cnns = [cnn_2, cnn_3]
    
    for idx, cnn in enumerate(cnns):
        q1_aligned, q2_aligned = soft_attention_alignment(q1_seq, q2_seq)
        q1_encoded = Concatenate()([q1_seq, q2_aligned, q1_encoded])
        q2_encoded = Concatenate()([q2_seq, q1_aligned, q2_encoded])            
        norm = BatchNormalization()

        q1_seq = cnn(q1_encoded)
        q1_seq = norm(q1_seq)
        q2_seq = cnn(q2_encoded)    
        q2_seq = norm(q2_seq)
    
    # Pooling
    q1_rep = apply_multiple(q1_seq, [GlobalAvgPool1D(), GlobalMaxPool1D(),])
    q2_rep = apply_multiple(q2_seq, [GlobalAvgPool1D(), GlobalMaxPool1D(),])    
    
    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    h_all = Concatenate()([q1_rep, q2_rep, q_diff, q_multi, ])
    h_all = Dropout(0.5)(h_all)
    h_all = Dense(256, activation='relu')(h_all)
    out_ = Dense(3, activation='softmax')(h_all)

    model = Model(inputs=[q1, q2, input_layer_3], outputs=out_)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6,), loss='categorical_crossentropy',
    metrics=['accuracy'])
    model.summary()
    return model

def get_raw_decomp_attn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
                           projection_dim=50, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=300, compare_dropout=0.2,
                           dense_dim=256, dense_dropout=0.2,
                           lr=1e-3, activation='relu'):
    
    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q1_c = Input(shape=(max_sequence_length, 11), name='first_sentences_char')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q2_c = Input(shape=(max_sequence_length, 11), name='second_sentences_char')
    input_layer_3 = Input(shape=(len(model_config.META_FEATURES),), name='mata-features', dtype="float32")
    
    embedding = Embedding(nb_words,
                            embedding_dim,
                            input_length=max_sequence_length,
                            trainable=True)
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.5)(q1_embed)
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.5)(q2_embed)
        
    # Deep view
    th = TimeDistributed(Highway(activation='relu'))
    th_2 = TimeDistributed(Highway(activation='relu'))
    
    q1_encoded = th(q1_embed,)
    #q1_encoded = Dropout(0.2)(q1_encoded)
    #q2_encoded = th(q2_embed,)
    #q2_encoded = Dropout(0.2)(q2_encoded)

    q2_encoded = th(q2_embed,)
    #q1_encoded = Dropout(0.2)(q1_encoded)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)    
    #r1_aligned, r2_aligned = soft_attention_alignment(r1_encoded, r2_encoded)
    
    # Compare Deep
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned),])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned),]) 
    
    compare_layers_d = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    
    q1_compare = time_distributed(q1_combined, compare_layers_d)
    q2_compare = time_distributed(q2_combined, compare_layers_d)

    # Compare Region
    #r1_combined = Concatenate()([r1_encoded, r2_aligned, submult(r1_encoded, r2_aligned)])
    #r2_combined = Concatenate()([r2_encoded, r1_aligned, submult(r2_encoded, r1_aligned)]) 
    
    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D(),])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D(),])

    # Dense meta featues
    meta_densed = BatchNormalization()(input_layer_3)
    meta_densed = Highway(activation='relu')(meta_densed)
    meta_densed = Dropout(0.2)(meta_densed)
    
    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    q_rep = Concatenate()([q1_rep, q2_rep])
    
    h_all = Concatenate()([q_diff, q_multi, q_rep,])
    #h_all = q_rep
    #h_all = Concatenate()([h_1, h_2, h_3, h_4])
    #h_all = BatchNormalization()(h_all)
    #h_all = Dropout(0.2)(h_all)
    #h_all = Highway(activation='relu')(h_all)
    #h_all = BatchNormalization()(h_all)
    h_all = Dropout(0.52)(h_all)

    #h_all = Highway(activation='relu')(h_all)
    #h_all = BatchNormalization()(h_all)
    #h_all = Dropout(0.2)(h_all)    
    
    #dense = BatchNormalization()(h_all)
    
    dense = Dense(dense_dim, activation=activation)(h_all)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    
    out_ = Dense(3, activation='softmax')(dense)
    
    model = Model(inputs=[q1, q2, input_layer_3], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    model.summary()
    return model


def get_sym_3d_cafe(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q1_c = Input(shape=(max_sequence_length, 11), name='first_sentences_char')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q2_c = Input(shape=(max_sequence_length, 11), name='second_sentences_char')
    input_layer_3 = Input(shape=(len(model_config.META_FEATURES),), name='mata-features', dtype="float32")

    embedding = Embedding(nb_words,
                            embedding_dim,
                            
                            input_length=max_sequence_length,
                            trainable=True)
    
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.5)(q1_embed)
    
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.5)(q2_embed)

    self_attention = SelfAttention(d_model=embedding_dim)

    # Deep view
    th = TimeDistributed(Highway(activation='relu'))

    q1_encoded = th(q1_embed,)    
    q2_encoded = th(q2_embed,)

    s1_encoded = self_attention(q1, q1_encoded)
    s2_encoded = self_attention(q2, q2_encoded)
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    
    # Compare Deep
    q1_combined1 = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned),])
    q2_combined1 = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned),])
    s1_combined1 = Concatenate()([q1_encoded, s1_encoded, submult(q1_encoded, s1_encoded),])
    s2_combined1 = Concatenate()([q2_encoded, s2_encoded, submult(q2_encoded, s2_encoded),])
    
    
    compare_layers_d = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(8, activation=activation),
        Dropout(compare_dropout),
    ]

    compare_layers_g = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(8, activation=activation),
        Dropout(compare_dropout),
    ]

    q1_compare = time_distributed(q1_combined1, compare_layers_d)
    q2_compare = time_distributed(q2_combined1, compare_layers_d)
    s1_compare = time_distributed(s1_combined1, compare_layers_g)
    s2_compare = time_distributed(s2_combined1, compare_layers_g)

    # Aggregate
    q1_encoded = Concatenate()([q1_encoded, q1_compare, s1_compare])
    q2_encoded = Concatenate()([q2_encoded, q2_compare, s2_compare])
    
    aggreate_rnn_1 = CuDNNGRU(76, return_sequences=True)    
    q1_aggreated = aggreate_rnn_1(q1_encoded)
    q1_aggreated = Dropout(0.2)(q1_aggreated)
    q2_aggreated = aggreate_rnn_1(q2_encoded)
    q2_aggreated = Dropout(0.2)(q2_aggreated)
    
    #q1_aggreated = fuse(q1_aggreated)
    #q2_aggreated = fuse(q2_aggreated)

    #q1_aggreated = Concatenate()([q1_encoded, q1_aggreated])
    #q2_aggreated = Concatenate()([q2_encoded, q2_aggreated])    
    
    # Pooling
    q1_rep = apply_multiple(q1_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(),])
    q2_rep = apply_multiple(q2_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(),])

    # Dense meta featues
    meta_densed = Highway(activation='relu')(input_layer_3)
    meta_densed = Dropout(0.2)(meta_densed)

    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    h_all1 = Concatenate()([q1_rep, q2_rep, q_diff, q_multi,])
    h_all1 = Dropout(0.5)(h_all1)
    
    dense = Dense(256, activation='relu')
    
    h_all = dense(h_all1)
    out_ = Dense(3, activation='softmax')(h_all)
    
    model = Model(inputs=[q1, q2, input_layer_3], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy',
    metrics=['accuracy'])
    model.summary()
    return model   

def get_sym_3d_meta_cafe(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q1_c = Input(shape=(max_sequence_length, 11), name='first_sentences_char')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q2_c = Input(shape=(max_sequence_length, 11), name='second_sentences_char')
    input_layer_3 = Input(shape=(len(model_config.META_FEATURES),), name='mata-features', dtype="float32")

    embedding = Embedding(nb_words,
                            embedding_dim,
                            
                            input_length=max_sequence_length,
                            trainable=True)
    
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.5)(q1_embed)
    
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.5)(q2_embed)

    self_attention = SelfAttention(d_model=embedding_dim)

    # Deep view
    th = TimeDistributed(Highway(activation='relu'))

    q1_encoded = th(q1_embed,)    
    q2_encoded = th(q2_embed,)

    s1_encoded = self_attention(q1, q1_encoded)
    s2_encoded = self_attention(q2, q2_encoded)
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    
    # Compare Deep
    q1_combined1 = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned),])
    q1_combined2 = Concatenate()([q2_aligned, q1_encoded, submult(q1_encoded, q2_aligned),])
    
    q2_combined1 = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned),])
    q2_combined2 = Concatenate()([q1_aligned, q2_encoded, submult(q2_encoded, q1_aligned),])
    
    s1_combined1 = Concatenate()([q1_encoded, s1_encoded, submult(q1_encoded, s1_encoded),])
    s1_combined2 = Concatenate()([s1_encoded, q1_encoded, submult(q1_encoded, s1_encoded),])
    
    s2_combined1 = Concatenate()([q2_encoded, s2_encoded, submult(q2_encoded, s2_encoded),])
    s2_combined2 = Concatenate()([s2_encoded, q2_encoded, submult(q2_encoded, s2_encoded),])
    
    
    compare_layers_d = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(8, activation=activation),
        Dropout(compare_dropout),
    ]

    compare_layers_g = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(8, activation=activation),
        Dropout(compare_dropout),
    ]

    q1_compare1 = time_distributed(q1_combined1, compare_layers_d)
    q1_compare2 = time_distributed(q1_combined2, compare_layers_d)
    q1_compare = Average()([q1_compare1, q1_compare2])

    q2_compare1 = time_distributed(q2_combined1, compare_layers_d)
    q2_compare2 = time_distributed(q2_combined2, compare_layers_d)
    q2_compare = Average()([q2_compare1, q2_compare2])
    
    s1_compare1 = time_distributed(s1_combined1, compare_layers_g)
    s1_compare2 = time_distributed(s1_combined2, compare_layers_g)
    s1_compare = Average()([s1_compare1, s1_compare2])
    
    s2_compare1 = time_distributed(s2_combined1, compare_layers_g)
    s2_compare2 = time_distributed(s2_combined2, compare_layers_g)
    s2_compare = Average()([s2_compare1, s2_compare2])

    # Aggregate
    q1_encoded = Concatenate()([q1_encoded, q1_compare, s1_compare])
    q2_encoded = Concatenate()([q2_encoded, q2_compare, s2_compare])
    
    aggreate_rnn_1 = CuDNNGRU(76, return_sequences=True)    
    q1_aggreated = aggreate_rnn_1(q1_encoded)
    q1_aggreated = Dropout(0.2)(q1_aggreated)
    q2_aggreated = aggreate_rnn_1(q2_encoded)
    q2_aggreated = Dropout(0.2)(q2_aggreated)
    
    #q1_aggreated = fuse(q1_aggreated)
    #q2_aggreated = fuse(q2_aggreated)

    #q1_aggreated = Concatenate()([q1_encoded, q1_aggreated])
    #q2_aggreated = Concatenate()([q2_encoded, q2_aggreated])    
    
    # Pooling
    q1_rep = apply_multiple(q1_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(),])
    q2_rep = apply_multiple(q2_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(),])

    # Dense meta featues
    meta_densed = Highway(activation='relu')(input_layer_3)
    meta_densed = Dropout(0.2)(meta_densed)

    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    h_all1 = Concatenate()([q1_rep, q2_rep, q_diff, q_multi, meta_densed])
    h_all2 = Concatenate()([q2_rep, q1_rep, q_diff, q_multi, meta_densed])
    h_all1 = Dropout(0.5)(h_all1)
    h_all2 = Dropout(0.5)(h_all2)
    
    dense = Dense(256, activation='relu')
    
    h_all1 = dense(h_all1)
    h_all2 = dense(h_all2)
    h_all = Average()([h_all1, h_all2])
    
    out_ = Dense(3, activation='softmax')(h_all)
    
    model = Model(inputs=[q1, q2, input_layer_3], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy',
    metrics=['accuracy'])
    model.summary()
    return model

def get_siamese_avcnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    embedding_layer = Embedding(nb_words,
        embedding_dim,
        
        input_length=max_sequence_length,
        trainable=True)
    filter_nums = 128
    # -- Dual input part --
    print("Prepareing Inputs")
    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q1_c = Input(shape=(max_sequence_length, 11), name='first_sentences_char')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q2_c = Input(shape=(max_sequence_length, 11), name='second_sentences_char')
    input_layer_3 = Input(shape=(len(model_config.META_FEATURES),), name='mata-features', dtype="float32")

    embedding_layer = Embedding(nb_words,
    embedding_dim,
    
    input_length=max_sequence_length,
    trainable=True)

    # -- Denstify --
    tdense = TimeDistributed(Highway(activation='relu'))

    embedded_1 = embedding_layer(q1)
    embedded_1 = SpatialDropout1D(0.5)(embedded_1)
    embedded_1 = tdense(embedded_1)

    embedded_2 = embedding_layer(q2)
    embedded_2 = SpatialDropout1D(0.5)(embedded_2)
    embedded_2 = tdense(embedded_2)

    conv_0 = Conv1D(filter_nums, 1, kernel_initializer="normal", padding="valid", activation="relu")
    conv_1 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")
    conv_2 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")

    avg_0 = GlobalAveragePooling1D()
    maxpool_0 = GlobalMaxPooling1D()

    attn_0 = AttentionWeightedAverage()
    attn_1 = AttentionWeightedAverage()
    attn_2 = AttentionWeightedAverage()

    sent_vecs = []

    for embedded in [embedded_1, embedded_2]:
        c0 = conv_0(embedded)
        c1 = conv_1(embedded)
        c2 = conv_2(embedded)

        a0 = attn_0(c0)
        a1 = attn_1(c1)
        a2 = attn_2(c2)

        g0 = avg_0(c0)
        g1 = avg_0(c1)
        g2 = avg_0(c2)

        m0 = maxpool_0(c0)
        m1 = maxpool_0(c1)
        m2 = maxpool_0(c2)

        v0_col = Concatenate()([a0, a1, a2])
        v1_col = Concatenate()([g0, g1, g2])
        v2_col = Concatenate()([m0, m1, m2])
        merged_tensor = Concatenate()([v0_col, v2_col, v1_col])
        #merged_tensor = Dropout(0.7)(merged_tensor)
        sent_vecs.append(merged_tensor)

    diff = Lambda(lambda t: K.abs(t[0] - t[1]), name='difference')(sent_vecs)
    multiply = Lambda(lambda t: t[0] * t[1], name='multiply')(sent_vecs)
    sent_vectors_pairs = concatenate(sent_vecs, axis=1)
    sent_vectors_pairs = concatenate([diff, multiply, sent_vectors_pairs], axis=1)
    sent_vectors_pairs = Dropout(0.7)(sent_vectors_pairs)

    # -- Dense meta featues --
    meta_features = BatchNormalization()(input_layer_3)
    meta_densed = Dense(96, activation='relu')(input_layer_3)
    meta_densed = Dropout(0.5)(meta_densed)
    meta_densed = BatchNormalization()(meta_densed)
    meta_densed = Dense(48, activation='relu')(meta_densed)
    meta_densed = Dropout(0.5)(meta_densed)

    all_outs = Concatenate()([meta_densed, sent_vectors_pairs])
    all_outs = sent_vectors_pairs

    output = Dense(units=300)(all_outs)
    output = Activation('relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)

    output = Dense(units=300)(output)
    output = Activation('relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)

    output = Dense(units=300)(output)
    output = Activation('relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)

    output = Dense(3, activation='softmax')(output)

    model = Model(inputs=[q1, q2, input_layer_3], outputs=output)
    adam_optimizer = optimizers.Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model   

def get_siamese_meta_avcnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    embedding_layer = Embedding(nb_words,
        embedding_dim,
        
        input_length=max_sequence_length,
        trainable=True)
    filter_nums = 128
    # -- Dual input part --
    print("Prepareing Inputs")
    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q1_c = Input(shape=(max_sequence_length, 11), name='first_sentences_char')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q2_c = Input(shape=(max_sequence_length, 11), name='second_sentences_char')
    input_layer_3 = Input(shape=(len(model_config.META_FEATURES),), name='mata-features', dtype="float32")

    embedding_layer = Embedding(nb_words,
    embedding_dim,
    
    input_length=max_sequence_length,
    trainable=True)

    # -- Denstify --
    tdense = TimeDistributed(Highway(activation='relu'))
    tdense_2 = TimeDistributed(Highway(activation='relu'))

    embedded_1 = embedding_layer(q1)
    embedded_1 = SpatialDropout1D(0.5)(embedded_1)
    embedded_1 = tdense(embedded_1)

    embedded_2 = embedding_layer(q2)
    embedded_2 = SpatialDropout1D(0.5)(embedded_2)
    embedded_2 = tdense(embedded_2)

    conv_0 = Conv1D(filter_nums, 1, kernel_initializer="normal", padding="valid", activation="relu")
    conv_1 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")
    conv_2 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")

    avg_0 = GlobalAveragePooling1D()
    maxpool_0 = GlobalMaxPooling1D()

    attn_0 = AttentionWeightedAverage()
    attn_1 = AttentionWeightedAverage()
    attn_2 = AttentionWeightedAverage()

    sent_vecs = []

    for embedded in [embedded_1, embedded_2]:
        c0 = conv_0(embedded)
        c1 = conv_1(embedded)
        c2 = conv_2(embedded)

        a0 = attn_0(c0)
        a1 = attn_1(c1)
        a2 = attn_2(c2)

        g0 = avg_0(c0)
        g1 = avg_0(c1)
        g2 = avg_0(c2)

        m0 = maxpool_0(c0)
        m1 = maxpool_0(c1)
        m2 = maxpool_0(c2)

        v0_col = Concatenate()([a0, a1, a2])
        v1_col = Concatenate()([g0, g1, g2])
        v2_col = Concatenate()([m0, m1, m2])
        merged_tensor = Concatenate()([v0_col, v2_col, v1_col])
        #merged_tensor = Dropout(0.7)(merged_tensor)
        sent_vecs.append(merged_tensor)

    diff = Lambda(lambda t: K.abs(t[0] - t[1]), name='difference')(sent_vecs)
    multiply = Lambda(lambda t: t[0] * t[1], name='multiply')(sent_vecs)
    sent_vectors_pairs = concatenate(sent_vecs, axis=1)
    sent_vectors_pairs = concatenate([diff, multiply, sent_vectors_pairs], axis=1)
    sent_vectors_pairs = Dropout(0.7)(sent_vectors_pairs)

    # -- Dense meta featues --
    meta_densed = Highway(activation='relu')(input_layer_3)
    meta_densed = Dropout(0.2)(meta_densed)

    all_outs = Concatenate()([meta_densed, sent_vectors_pairs])

    output = Dense(units=300)(all_outs)
    output = Activation('relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)

    output = Dense(units=300)(output)
    output = Activation('relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)

    output = Dense(units=300)(output)
    output = Activation('relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)

    output = Dense(3, activation='softmax')(output)

    model = Model(inputs=[q1, q2, input_layer_3], outputs=output)
    adam_optimizer = optimizers.Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model

def get_4way_cafe(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=300, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q1_c = Input(shape=(max_sequence_length, 11), name='first_sentences_char')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q2_c = Input(shape=(max_sequence_length, 11), name='second_sentences_char')
    input_layer_3 = Input(shape=(len(model_config.META_FEATURES),), name='mata-features', dtype="float32")

    embedding = Embedding(nb_words,
                            embedding_dim,
                            input_length=max_sequence_length,
                            trainable=True)
    
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.5)(q1_embed)
    
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.5)(q2_embed)

    self_attention = SelfAttention(d_model=embedding_dim)

    # Deep view
    th = TimeDistributed(Highway(activation='relu'))

    q1_encoded = th(q1_embed,)    
    q2_encoded = th(q2_embed,)

    s1_encoded = self_attention(q1, q1_encoded)
    s2_encoded = self_attention(q2, q2_encoded)
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    
    # Compare Deep
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned),])    
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned),])
    s1_combined = Concatenate()([q1_encoded, s1_encoded, submult(q1_encoded, s1_encoded),])
    s2_combined = Concatenate()([q2_encoded, s2_encoded, submult(q2_encoded, s2_encoded),])
        
    compare_layers_d = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(10, activation=activation),
        Dropout(compare_dropout),
    ]

    compare_layers_g = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(10, activation=activation),
        Dropout(compare_dropout),
    ]

    q1_compare = time_distributed(q1_combined, compare_layers_d)
    q2_compare = time_distributed(q2_combined, compare_layers_d)

    s1_compare = time_distributed(s1_combined, compare_layers_g)
    s2_compare = time_distributed(s2_combined, compare_layers_g)
    
    # Aggregate
    q1_encoded = Concatenate()([q1_encoded, q1_compare, s1_compare])
    q2_encoded = Concatenate()([q2_encoded, q2_compare, s2_compare])
    
    aggreate_rnn_1 = CuDNNGRU(72, return_sequences=True)    
    q1_aggreated = aggreate_rnn_1(q1_encoded)
    q1_aggreated = Dropout(0.2)(q1_aggreated)
    q2_aggreated = aggreate_rnn_1(q2_encoded)
    q2_aggreated = Dropout(0.2)(q2_aggreated)
    
    #q1_aggreated = fuse(q1_aggreated)
    #q2_aggreated = fuse(q2_aggreated)

    #q1_aggreated = Concatenate()([q1_encoded, q1_aggreated])
    #q2_aggreated = Concatenate()([q2_encoded, q2_aggreated])    
    
    attention = AttentionWeightedAverage()
    last = Lambda(lambda v: v[:, -1, :])
    
    # Pooling
    q1_rep = apply_multiple(q1_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(), attention, last])
    q2_rep = apply_multiple(q2_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(), attention, last])

    # Dense meta featues
    meta_densed = Highway(activation='relu')(input_layer_3)
    meta_densed = Dropout(0.2)(meta_densed)

    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    h_all = Concatenate()([q1_rep, q2_rep, q_diff, q_multi,])
    h_all = Dropout(0.5)(h_all)
    h_all = Dense(300, activation='relu')(h_all)    
    out_ = Dense(3, activation='softmax')(h_all)
    
    model = Model(inputs=[q1, q2, input_layer_3], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy',
    metrics=['accuracy'])
    model.summary()
    return model

def get_4way_meta_cafe(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=300, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q1_c = Input(shape=(max_sequence_length, 11), name='first_sentences_char')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q2_c = Input(shape=(max_sequence_length, 11), name='second_sentences_char')
    input_layer_3 = Input(shape=(len(model_config.META_FEATURES),), name='mata-features', dtype="float32")

    embedding = Embedding(nb_words,
                            embedding_dim,
                            input_length=max_sequence_length,
                            trainable=True)
    
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.5)(q1_embed)
    
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.5)(q2_embed)

    self_attention = SelfAttention(d_model=embedding_dim)

    # Deep view
    th = TimeDistributed(Highway(activation='relu'))

    q1_encoded = th(q1_embed,)    
    q2_encoded = th(q2_embed,)

    s1_encoded = self_attention(q1, q1_encoded)
    s2_encoded = self_attention(q2, q2_encoded)
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    
    # Compare Deep
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned),])    
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned),])
    s1_combined = Concatenate()([q1_encoded, s1_encoded, submult(q1_encoded, s1_encoded),])
    s2_combined = Concatenate()([q2_encoded, s2_encoded, submult(q2_encoded, s2_encoded),])
        
    compare_layers_d = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(10, activation=activation),
        Dropout(compare_dropout),
    ]

    compare_layers_g = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(10, activation=activation),
        Dropout(compare_dropout),
    ]

    q1_compare = time_distributed(q1_combined, compare_layers_d)
    q2_compare = time_distributed(q2_combined, compare_layers_d)

    s1_compare = time_distributed(s1_combined, compare_layers_g)
    s2_compare = time_distributed(s2_combined, compare_layers_g)
    
    # Aggregate
    q1_encoded = Concatenate()([q1_encoded, q1_compare, s1_compare])
    q2_encoded = Concatenate()([q2_encoded, q2_compare, s2_compare])
    
    aggreate_rnn_1 = CuDNNGRU(72, return_sequences=True)    
    q1_aggreated = aggreate_rnn_1(q1_encoded)
    q1_aggreated = Dropout(0.2)(q1_aggreated)
    q2_aggreated = aggreate_rnn_1(q2_encoded)
    q2_aggreated = Dropout(0.2)(q2_aggreated)
    
    #q1_aggreated = fuse(q1_aggreated)
    #q2_aggreated = fuse(q2_aggreated)

    #q1_aggreated = Concatenate()([q1_encoded, q1_aggreated])
    #q2_aggreated = Concatenate()([q2_encoded, q2_aggreated])    
    
    attention = AttentionWeightedAverage()
    last = Lambda(lambda v: v[:, -1, :])
    
    # Pooling
    q1_rep = apply_multiple(q1_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(), attention, last])
    q2_rep = apply_multiple(q2_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(), attention, last])

    # Dense meta featues
    meta_densed = Highway(activation='relu')(input_layer_3)
    meta_densed = Dropout(0.2)(meta_densed)

    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    h_all = Concatenate()([q1_rep, q2_rep, q_diff, q_multi, meta_densed])
    h_all = Dropout(0.5)(h_all)
    h_all = Dense(300, activation='relu')(h_all)    
    out_ = Dense(3, activation='softmax')(h_all)
    
    model = Model(inputs=[q1, q2, input_layer_3], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy',
    metrics=['accuracy'])
    model.summary()
    return model

def get_dense_meta_cnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q1_c = Input(shape=(max_sequence_length, 11), name='first_sentences_char')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q2_c = Input(shape=(max_sequence_length, 11), name='second_sentences_char')
    input_layer_3 = Input(shape=(len(model_config.META_FEATURES),), name='mata-features', dtype="float32")

    embedding = Embedding(nb_words,
                            embedding_dim,
                            
                            input_length=max_sequence_length,
                            trainable=True)
    
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.5)(q1_embed)
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.5)(q2_embed)

    th = TimeDistributed(Highway(activation='relu'))
    
    q1_encoded = th(q1_embed,)    
    q2_encoded = th(q2_embed,)
    
    cnn_init = Conv1D(32, 1, strides=1, padding='same', activation='relu')

    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    q1_encoded = Concatenate()([q2_aligned, q1_encoded])
    q2_encoded = Concatenate()([q1_aligned, q2_encoded])     
    norm = BatchNormalization()

    q1_seq = cnn_init(q1_encoded)
    q1_seq = norm(q1_seq)
    q2_seq = cnn_init(q2_encoded)    
    q2_seq = norm(q2_seq)
    
    cnn_2 = Conv1D(32, 3, strides=1, padding='same', activation='relu')
    cnn_3 = Conv1D(32, 1, strides=1, padding='same', activation='relu')
    cnn_4 = Conv1D(32, 3, strides=1, padding='same', activation='relu')
    cnn_5 = Conv1D(32, 1, strides=1, padding='same', activation='relu') 

    cnns = [cnn_2, cnn_3]
    #rnn_6 = CuDNNGRU(6, return_sequences=True,)
    
    for idx, cnn in enumerate(cnns):
        q1_aligned, q2_aligned = soft_attention_alignment(q1_seq, q2_seq)
        q1_encoded = Concatenate()([q1_seq, q2_aligned, q1_encoded])
        q2_encoded = Concatenate()([q2_seq, q1_aligned, q2_encoded])            
        norm = BatchNormalization()

        q1_seq = cnn(q1_encoded)
        q1_seq = norm(q1_seq)
        q2_seq = cnn(q2_encoded)    
        q2_seq = norm(q2_seq)
    
    # Pooling
    attn = AttentionWeightedAverage()
    q1_rep = apply_multiple(q1_seq, [GlobalAvgPool1D(), GlobalMaxPool1D(), attn])
    q2_rep = apply_multiple(q2_seq, [GlobalAvgPool1D(), GlobalMaxPool1D(), attn])    
    
    # Classifier
    meta_densed = Highway(activation='relu')(input_layer_3)
    meta_densed = Dropout(0.2)(meta_densed)
    
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    h_all = Concatenate()([q1_rep, q2_rep, q_diff, q_multi, meta_densed])
    h_all = Dropout(0.5)(h_all)
    h_all = Dense(256, activation='relu')(h_all)
    out_ = Dense(3, activation='softmax')(h_all)

    model = Model(inputs=[q1, q2, input_layer_3], outputs=out_)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6,), loss='categorical_crossentropy',
    metrics=['accuracy'])
    model.summary()
    return model

class ModelManager(object):
    
    def __init__(self, *args, **kwargs):
        
        # Note: should refactor to dict
        self.models_tag = [
            'RawDecomposableAttn',
            'MetaDecomposableAttn',

            'Symmetric3D',
            'Symmetric3DMeta',  # SymNet Family

            'DenseCNN',
            'DenseMetaCNN', # DenseCNN Family

            'DeepAVCNN',
            'DeepMetaAVCNN',
            
            '4way_net',
            '4way_metaNet',
        ]

        self.models_checkpoints_pathes = [
            '../model/model_checkpoint/DecomposableAttn/RawDecomposableAttn-NoScale-Patience40',
            '../model/model_checkpoint/DecomposableAttn/DecomposableAttn-NoScale-Patience40',

            "../model/model_checkpoint/Syms/Symetric-SELU3DCafe-AllNoScale-Patience30",
            "../model/model_checkpoint/Syms/Symetric-SELU3DMetaCafe-AllNoScale-Patience30", # SymNet Family

            "../model/model_checkpoint/ScaledDenseCNN/DenseCNN-Patience60",
            '../model/model_checkpoint/ScaledDenseCNN/DenseMetaCNN-Patience60', # DenseCNN Family

            "../model/model_checkpoint/DeepAVCNN/DeepAVCNN-NoScale-Patience50",
            "../model/model_checkpoint/DeepAVCNN/DeepMetaAVCNN-NoScale-Patience50",

            "../model/model_checkpoint/4Way/4Way-NoClassScale-NoValidScale-Patience50",
            "../model/model_checkpoint/4Way/4WayMeta-NoClassScale-NoValidScale-Patience50",
            
        ]

        self.submit_predix = [
            "RawDecomposableAttn-NoScale-Patience40",
            "DecomposableAttn-NoScale-Patience40",

            "SymmetricSelu3DCafe-AllNoScale",
            "SymmetricSelu3DMetaCafe-ScaleClass",  # SymNet Family

            "DenseCNN-ScaleClass-ScaleValid",
            "DenseMetaCNN-ScaleClass",

            "DeepAVCNN-NoScale",
            "DeepMetaAVCNN-NoScale",

            "4Way-NoScale",
            "Meta4Way-NoScale",
        ]

        self.model_funcs = [
            get_raw_decomp_attn,
            get_decomp_attn,

            get_sym_3d_cafe,
            get_sym_3d_meta_cafe, # Symmetric net family

            get_dense_cnn,
            get_dense_meta_cnn,

            get_siamese_avcnn,
            get_siamese_meta_avcnn, # avcnn family

            get_4way_cafe,
            get_4way_meta_cafe, # 4way family
        ]

        self.model_patiences = [
            15,
            13,

            10,
            10,

            20,
            20,

            15,
            15,

            10,
            10,
        ]

        self.model_class_weights = [True] * 10
        self.model_scale_sample_weights = [True] * 10

    #def agent_func(NB_WORDS, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, OUT_SIZE):
    #    return 
        
