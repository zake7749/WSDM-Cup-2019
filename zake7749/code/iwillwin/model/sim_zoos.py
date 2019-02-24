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

def get_dense_cnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    meta_features_input = Input(shape=(36,), name='mata-features')
    
    
    embedding = Embedding(nb_words, embedding_dim,
                          weights=[embedding_matrix],
                          input_length=max_sequence_length,
                          trainable=False)
    
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.2)(q1_embed)
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.2)(q2_embed)

    th = TimeDistributed(Highway(activation='relu'))
    
    q1_encoded = th(q1_embed,)    
    q2_encoded = th(q2_embed,)
    
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    q1_encoded = Concatenate()([q2_aligned, q1_encoded])
    q2_encoded = Concatenate()([q1_aligned, q2_encoded])  
    
    cnn_init = Conv1D(42, 1, strides=1, padding='same', activation='relu')
    q1_seq = cnn_init(q1_encoded)
    q2_seq = cnn_init(q2_encoded)
    
    cnns = [Conv1D(42, 3, strides=1, padding='same', activation='relu') for i in range(3)]
    trans = [Conv1D(32, 1, strides=1, padding='same', activation='relu') for i in range(3)]
    
    for idx, cnn in enumerate(cnns):
        q1_aligned, q2_aligned = soft_attention_alignment(q1_seq, q2_seq)
        q1_encoded = Concatenate()([q1_seq, q2_aligned, q1_encoded])
        q2_encoded = Concatenate()([q2_seq, q1_aligned, q2_encoded])            
        q1_seq = cnn(q1_encoded)
        q2_seq = cnn(q2_encoded)    
    
    attn = AttentionWeightedAverage()
    
    
    q1_rep = apply_multiple(q1_encoded, [GlobalAvgPool1D(), GlobalMaxPool1D(), attn])
    q2_rep = apply_multiple(q2_encoded, [GlobalAvgPool1D(), GlobalMaxPool1D(), attn])    
    
    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    h_all = Concatenate()([q1_rep, q2_rep, q_diff, q_multi,])
    h_all = Dropout(0.5)(h_all)
  
    h_all = Dense(128, activation='relu')(h_all)
    out_ = Dense(3, activation='softmax')(h_all)

    model = Model(inputs=[q1, q2, meta_features_input], outputs=out_)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6, clipnorm=1), loss='categorical_crossentropy',
    metrics=['accuracy', weighted_accuracy])
    model.summary()
    return model

def get_multiwindow_cnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    meta_features_input = Input(shape=(36,), name='mata-features')
    
    embedding = Embedding(nb_words, embedding_dim,
                          weights=[embedding_matrix],
                          input_length=max_sequence_length,
                          trainable=False)
    
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.2)(q1_embed)
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.2)(q2_embed)

    th = TimeDistributed(Highway(activation='relu'))
    
    q1_encoded = th(q1_embed,)    
    q2_encoded = th(q2_embed,)
    
    q1_in = q1_encoded
    q2_in = q2_encoded
    
    nb_filters = 64
    
    for i in range(1, 5):
        tanh_conv = Conv1D(nb_filters, i, padding='same', activation='tanh')
        sigm_conv = Conv1D(nb_filters, i, padding='same', activation='sigmoid')
        res_conv = Conv1D(nb_filters, i, padding='same', activation='relu')
        drop = Dropout(0.1)
        
        q1_t = tanh_conv(q1_in)
        q1_s = sigm_conv(q1_in)
        q1_x = Multiply()([q1_t, q1_s])
        
        res_q1 = res_conv(q1_x)
        res_q1 = drop(res_q1)
        q1_encoded = Concatenate()([q1_encoded, q1_x])
        
        q2_t = tanh_conv(q2_in)
        q2_s = sigm_conv(q2_in)       
        q2_x = Multiply()([q2_t, q2_s])
        
        res_q2 = res_conv(q2_x)
        res_q2 = drop(res_q2)
        q2_encoded = Concatenate()([q2_encoded, q2_x])

    # Align after align
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    q1_encoded = Concatenate()([q1_encoded, q2_aligned,])
    q2_encoded = Concatenate()([q2_encoded, q1_aligned,])
   
    attn = AttentionWeightedAverage()
    q1_rep = apply_multiple(q1_encoded, [GlobalAvgPool1D(), GlobalMaxPool1D(), attn,])
    q2_rep = apply_multiple(q2_encoded, [GlobalAvgPool1D(), GlobalMaxPool1D(), attn,])    
    
    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    h_all = Concatenate()([q1_rep, q2_rep, q_diff, q_multi,])
    h_all = Dropout(0.2)(h_all)
    out_ = Dense(3, activation='softmax')(h_all)

    model = Model(inputs=[q1, q2, meta_features_input], outputs=out_)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6, clipnorm=1.5), loss='categorical_crossentropy',
    metrics=['accuracy', weighted_accuracy])
    model.summary()
    return model

def get_char_dense_cnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    meta_features_input = Input(shape=(36,), name='mata-features')
    
    q1_exact_match = Input(shape=(max_sequence_length,), name='first_exact_match')
    q2_exact_match = Input(shape=(max_sequence_length,), name='second_exact_match')
    
    embedding = Embedding(nb_words, 150,
                          weights=[embedding_matrix],
                          input_length=max_sequence_length,
                          trainable=False)
    
    flex_embedding = Embedding(nb_words, 20,
                      input_length=max_sequence_length,
                      trainable=True)
    
    em_embeddings = Reshape((max_sequence_length, 1))
    
    q1_embed = Concatenate()([embedding(q1), em_embeddings(q1_exact_match),])
    q1_encoded = SpatialDropout1D(0.2)(q1_embed)
    
    q2_embed = Concatenate()([embedding(q2), em_embeddings(q2_exact_match),])
    q2_encoded = SpatialDropout1D(0.2)(q2_embed)
    nb_filters = 64
    
    cnns = [Conv1D(64, 1, strides=1, padding='same', activation='relu') for i in range(3)]
    gates_cnns = [Conv1D(nb_filters, 3, dilation_rate=1, padding='same', activation='tanh') for i in range(3)]
    sigm_cnns = [Conv1D(nb_filters, 3, dilation_rate=1, padding='same', activation='sigmoid') for i in range(3)]
    
    for i in range(len(cnns)):
        drop = Dropout(0.1)
        q1_t = gates_cnns[i](q1_encoded)
        q2_t = gates_cnns[i](q2_encoded)    
        
        q1_s = sigm_cnns[i](q1_encoded)
        q2_s = sigm_cnns[i](q2_encoded)        
        
        q1_x = Multiply()([q1_t, q1_s])
        q1_x = cnns[i](q1_x)
        q1_x = drop(q1_x)
        
        q2_x = Multiply()([q2_t, q2_s])
        q2_x = cnns[i](q2_x)
        q2_x = drop(q2_x)

        q1_aligned, q2_aligned = soft_attention_alignment(q1_x, q2_x)
        q1_encoded = Concatenate()([q1_x, q2_aligned, q1_encoded])
        q2_encoded = Concatenate()([q2_x, q1_aligned, q2_encoded]) 
    
    attn = AttentionWeightedAverage()
    q1_rep = apply_multiple(q1_encoded, [GlobalAvgPool1D(), GlobalMaxPool1D(), attn])
    q2_rep = apply_multiple(q2_encoded, [GlobalAvgPool1D(), GlobalMaxPool1D(), attn])    
    
    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    h_all = Concatenate()([q1_rep, q2_rep, q_diff, q_multi,])
    h_all = Dropout(0.2)(h_all)
    h_all = BatchNormalization()(h_all)  
    h_all = Dense(64, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1e-6))(h_all)
    out_ = Dense(3, activation='softmax')(h_all)

    model = Model(inputs=[q1, q2, meta_features_input, q1_exact_match, q2_exact_match], outputs=out_)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6, clipnorm=1.5), loss='categorical_crossentropy',
    metrics=['accuracy', weighted_accuracy])
    model.summary()
    return model

def get_darnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')

    q1_exact_match = Input(shape=(max_sequence_length,), name='first_exact_match')
    q2_exact_match = Input(shape=(max_sequence_length,), name='second_exact_match')    
    input_layer_3 = Input(shape=(36,), name='mata-features', dtype="float32")
    
    embedding = Embedding(nb_words, embedding_dim,
                          weights=[embedding_matrix],
                          input_length=max_sequence_length,
                          trainable=False)
    
    em_embeddings = Embedding(2, 1,
                     input_length=max_sequence_length,
                     trainable=True)   
    
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.1)(q1_embed)
    
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.1)(q2_embed)

    th = TimeDistributed(Highway(activation='relu'))
    q1_embed = Dropout(0.1)(th(q1_embed,))
    q2_embed = Dropout(0.1)(th(q2_embed,))    
    
    rnns = [Bidirectional(CuDNNGRU(42, return_sequences=True)) for i in range(3)]
    
    q1_res = []
    q2_res = []
    
    
    for idx, rnn in enumerate(rnns):
        q1_seq = rnn(q1_embed)
        q1_seq = Dropout(0.15)(q1_seq)
        q2_seq = rnn(q2_embed)
        q2_seq = Dropout(0.15)(q2_seq)
        q1_aligned, q2_aligned = soft_attention_alignment(q1_seq, q2_seq)
        
        q1_res.append(q2_aligned)
        q1_res.append(q1_seq)
        
        q2_res.append(q1_aligned)
        q2_res.append(q2_seq)
        
        q1_embed = Concatenate()([q1_embed, q1_seq, q2_aligned,])
        q2_embed = Concatenate()([q2_embed, q2_seq, q1_aligned,])

    q1_res = Concatenate()(q1_res)
    q2_res = Concatenate()(q2_res)
    
    attn = AttentionWeightedAverage()
    q1_rep = apply_multiple(q1_embed, [GlobalAvgPool1D(), GlobalMaxPool1D(), attn])
    q2_rep = apply_multiple(q2_embed, [GlobalAvgPool1D(), GlobalMaxPool1D(), attn])   
    
    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    h_all = Concatenate()([q1_rep, q2_rep, q_diff, q_multi,])
    h_all = Dropout(0.35)(h_all)
    h_all = Dense(300, activation='relu')(h_all)
    out_ = Dense(3, activation='softmax')(h_all)

    model = Model(inputs=[q1, q2, input_layer_3, q1_exact_match, q2_exact_match], outputs=out_)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6, clipvalue=1.5), loss='categorical_crossentropy',
    metrics=['accuracy', weighted_accuracy])
    model.summary()
    return model

def get_char_darnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q1_exact_match = Input(shape=(max_sequence_length,), name='first_exact_match')
    q2_exact_match = Input(shape=(max_sequence_length,), name='second_exact_match')
    
    input_layer_3 = Input(shape=(36,), name='mata-features', dtype="float32")

    embedding = Embedding(nb_words, 150,
                          weights=[embedding_matrix],
                          input_length=max_sequence_length,
                          trainable=False)
 
    em_embeddings = Embedding(2, 1,
                     input_length=max_sequence_length,
                     trainable=True)   
    
    q1_embed = Concatenate()([embedding(q1), em_embeddings(q1_exact_match)])
    q1_embed = SpatialDropout1D(0.1)(q1_embed)
    
    q2_embed = Concatenate()([embedding(q2), em_embeddings(q2_exact_match)])
    q2_embed = SpatialDropout1D(0.1)(q2_embed)
    
    rnns = [CuDNNGRU(42, return_sequences=True) for i in range(3)]
    
    q1_res = []
    q2_res = []
    
    
    for idx, rnn in enumerate(rnns):
        q1_seq = rnn(q1_embed)
        q1_seq = Dropout(0.1)(q1_seq)
        q2_seq = rnn(q2_embed)
        q2_seq = Dropout(0.1)(q2_seq)
        q1_aligned, q2_aligned = soft_attention_alignment(q1_seq, q2_seq)
        
        q1_res.append(q2_aligned)
        q1_res.append(q1_seq)
        
        q2_res.append(q1_aligned)
        q2_res.append(q2_seq)
        
        q1_embed = Concatenate()([q1_seq, q2_aligned, q1_embed])
        q2_embed = Concatenate()([q2_seq, q1_aligned, q2_embed])            
        
    # Pooling
    #q1_rep = Flatten()(capsule_pooling(q1_encoded))
    #q2_rep = Flatten()(capsule_pooling(q2_encoded))
    
    q1_res = Concatenate()(q1_res)
    q2_res = Concatenate()(q2_res)
    
    q1_rep = apply_multiple(q1_res, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_res, [GlobalAvgPool1D(), GlobalMaxPool1D()])    
    
    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    h_all = Concatenate()([q1_rep, q2_rep, q_diff, q_multi,])
    h_all = Dropout(0.1)(h_all)
    h_all = Dense(256, activation='relu')(h_all)
    out_ = Dense(3, activation='softmax')(h_all)

    model = Model(inputs=[q1, q2, input_layer_3, q1_exact_match, q2_exact_match], outputs=out_)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6,), loss='categorical_crossentropy',
    metrics=['accuracy', weighted_accuracy])
    model.summary()
    return model

def get_ESIM(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q1_exact_match = Input(shape=(max_sequence_length,), name='first_exact_match')
    q2_exact_match = Input(shape=(max_sequence_length,), name='second_exact_match')
    
    input_layer_3 = Input(shape=(36,), name='mata-features', dtype="float32")
    
    embedding = Embedding(nb_words, embedding_dim,
                          weights=[embedding_matrix],
                          input_length=max_sequence_length,
                          trainable=False)
    
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.1)(q1_embed)
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.1)(q2_embed)

    batch_norm = BatchNormalization(axis=-1)
    q1_embed = batch_norm(q1_embed,)
    q2_embed = batch_norm(q2_embed,)  
    
    aggreation_gru = Bidirectional(CuDNNLSTM(100, return_sequences=True))
 
    q1_seq = aggreation_gru(q1_embed)
    q2_seq = aggreation_gru(q2_embed)
        
    q1_aligned, q2_aligned = soft_attention_alignment(q1_seq, q2_seq)
    
    q1_vec = Concatenate()([q1_seq, q2_aligned, substract(q1_seq, q2_aligned), Multiply()([q1_seq, q2_aligned])])
    q2_vec = Concatenate()([q2_seq, q1_aligned, substract(q2_seq, q1_aligned), Multiply()([q2_seq, q1_aligned])])
    
    compare_gru = Bidirectional(CuDNNLSTM(100, return_sequences=True))
    
    q1_rep = compare_gru(q1_vec)
    q2_rep = compare_gru(q2_vec)
    
    q1_rep = apply_multiple(q1_rep, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_rep, [GlobalAvgPool1D(), GlobalMaxPool1D()])    
    
    h_all = Concatenate()([q1_rep, q2_rep])
    h_all = BatchNormalization()(h_all)
    
    h_all = Dense(256, activation='elu')(h_all)
    h_all = BatchNormalization()(h_all)
    h_all = Dropout(0.5)(h_all)
    
    h_all = Dense(256, activation='elu')(h_all)
    h_all = BatchNormalization()(h_all)
    h_all = Dropout(0.5)(h_all)
   
    out_ = Dense(3, activation='softmax')(h_all)
    
    model = Model(inputs=[q1, q2, input_layer_3, q1_exact_match, q2_exact_match], outputs=out_)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6, clipnorm=1.5,), loss='categorical_crossentropy',
    metrics=['accuracy', weighted_accuracy])
    model.summary()
    return model

def get_char_ESIM(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q1_exact_match = Input(shape=(max_sequence_length,), name='first_exact_match')
    q2_exact_match = Input(shape=(max_sequence_length,), name='second_exact_match')
    
    input_layer_3 = Input(shape=(36,), name='mata-features', dtype="float32")

    #input_encoded = BatchNormalization()(input_layer_3)
    input_encoded = Dense(2016, activation='elu')(input_layer_3)
    input_encoded = Dropout(0.25)(input_encoded)
    
    embedding = Embedding(nb_words, 150,
                          weights=[embedding_matrix],
                          input_length=max_sequence_length,
                          trainable=False)
 
    em_embeddings = Embedding(2, 1,
                     input_length=max_sequence_length,
                     trainable=True)   
    
    #q1_embed = Concatenate()([embedding(q1), em_embeddings(q1_exact_match)])
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.1)(q1_embed)
    
    #q2_embed = Concatenate()([embedding(q2), em_embeddings(q2_exact_match)])
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.1)(q2_embed)

    batch_norm = BatchNormalization(axis=-1)
    q1_embed = batch_norm(q1_embed)
    q2_embed = batch_norm(q2_embed)
    
    aggreation_gru = Bidirectional(CuDNNLSTM(72, return_sequences=True))
 
    q1_seq = aggreation_gru(q1_embed)
    q2_seq = aggreation_gru(q2_embed)
        
    q1_aligned, q2_aligned = soft_attention_alignment(q1_seq, q2_seq)
    q1_vec = Concatenate()([q1_seq, q2_aligned, substract(q1_seq, q2_aligned), Multiply()([q1_seq, q2_aligned])])
    q2_vec = Concatenate()([q2_seq, q1_aligned, substract(q2_seq, q1_aligned), Multiply()([q2_seq, q1_aligned])])
    
    compare_gru = Bidirectional(CuDNNLSTM(72, return_sequences=True))
    
    q1_rep = compare_gru(q1_vec)
    q2_rep = compare_gru(q2_vec)
    
    q1_rep = apply_multiple(q1_rep, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_rep, [GlobalAvgPool1D(), GlobalMaxPool1D()])    
    
    h_all = Concatenate()([q1_rep, q2_rep])
    h_all = BatchNormalization()(h_all)
    
    h_all = Dense(256, activation='elu')(h_all)
    h_all = BatchNormalization()(h_all)
    h_all = Dropout(0.2)(h_all)
    
    h_all = Dense(256, activation='elu')(h_all)
    h_all = BatchNormalization()(h_all)
    h_all = Dropout(0.2)(h_all)
    
    out_ = Dense(3, activation='softmax')(h_all)
    
    model = Model(inputs=[q1, q2, input_layer_3, q1_exact_match, q2_exact_match], outputs=out_)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6, clipnorm=1.5, amsgrad=True), loss='categorical_crossentropy',
    metrics=['accuracy', weighted_accuracy])
    model.summary()
    return model

# should add a util.py for scoring
def weighted_accuracy(y_true, y_pred):
    weight = np.array([[1/16, 1/15, 1/5]])
    norm = [(1/16) + (1/15) + (1/5)]
    weight_mask = weight * y_true
    label_weights = K.max(K.cast(weight_mask, 'float32'), axis=-1)
    
    true_label = K.argmax(y_true, axis=-1)
    pred_label = K.argmax(y_pred, axis=-1)
    
    res = K.cast(K.equal(true_label, pred_label), tf.float32) * label_weights / K.sum(label_weights)
    res = K.sum(res)
    return res

def numpy_weighted_accuracy(y_true, y_pred):
    weight = np.array([[1/16, 1/15, 1/5]])
    norm = [(1/16) + (1/15) + (1/5)]
    weight_mask = weight * y_true
    
    y_pred = (y_pred > 0.5).astype(int)
    
    res = np.equal(y_pred, y_true) * weight_mask / np.sum(weight_mask)
    res = np.sum(res)
    return res   

def get_decomposable_attention(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q1_exact_match = Input(shape=(max_sequence_length,), name='first_exact_match')
    q2_exact_match = Input(shape=(max_sequence_length,), name='second_exact_match')    
    input_layer_3 = Input(shape=(36,), name='mata-features', dtype="float32")
    
    embedding = Embedding(nb_words, embedding_dim,
                          weights=[embedding_matrix],
                          input_length=max_sequence_length,
                          trainable=False)
    
    em_embeddings = Embedding(2, 1,
                     input_length=max_sequence_length,
                     trainable=True)   
    
    #q1_embed = Concatenate()([embedding(q1), em_embeddings(q1_exact_match)])
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.1)(q1_embed)
    
    #q2_embed = Concatenate()([embedding(q2), em_embeddings(q2_exact_match)])
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.1)(q2_embed)

    th = TimeDistributed(Highway(activation='relu'))
    q1_embed = th(q1_embed)
    q2_embed = th(q2_embed)
        
    q1_aligned, q2_aligned = soft_attention_alignment(q1_embed, q2_embed)
    q1_vec = Concatenate()([q1_embed, q2_aligned, substract(q1_embed, q2_aligned), Multiply()([q1_embed, q2_aligned])])
    q2_vec = Concatenate()([q2_embed, q1_aligned, substract(q2_embed, q1_aligned), Multiply()([q2_embed, q1_aligned])])
    
    dense_compares = [
        Dense(300, activation='elu'),
        Dropout(0.2),
        Dense(200, activation='elu'),
        Dropout(0.2),
    ]

    q1_compared = time_distributed(q1_vec, dense_compares)
    q2_compared = time_distributed(q2_vec, dense_compares)
    
    q1_rep = apply_multiple(q1_compared, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compared, [GlobalAvgPool1D(), GlobalMaxPool1D()])    
    
    h_all = Concatenate()([q1_rep, q2_rep])
    h_all = BatchNormalization()(h_all)
    
    h_all = Dense(256, activation='elu')(h_all)
    h_all = Dropout(0.2)(h_all)
    h_all = BatchNormalization()(h_all)

    h_all = Dense(256, activation='elu')(h_all)
    h_all = Dropout(0.2)(h_all)
    h_all = BatchNormalization()(h_all)    
    
    out_ = Dense(3, activation='softmax')(h_all)
    
    model = Model(inputs=[q1, q2, input_layer_3, q1_exact_match, q2_exact_match], outputs=out_)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6, clipnorm=1.5, amsgrad=True), loss='categorical_crossentropy',
    metrics=['accuracy', weighted_accuracy])
    model.summary()
    return model

def get_char_decomposable_attention(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size,
    projection_dim=50, projection_hidden=0, projection_dropout=0.2,
    compare_dim=288, compare_dropout=0.2,
    dense_dim=50, dense_dropout=0.2,
    lr=1e-3, activation='relu'):

    q1 = Input(shape=(max_sequence_length,), name='first_sentences')
    q2 = Input(shape=(max_sequence_length,), name='second_sentences')
    q1_exact_match = Input(shape=(max_sequence_length,), name='first_exact_match')
    q2_exact_match = Input(shape=(max_sequence_length,), name='second_exact_match')

    input_layer_3 = Input(shape=(36,), name='mata-features', dtype="float32")

    #input_encoded = BatchNormalization()(input_layer_3)
    input_encoded = Dense(2016, activation='elu')(input_layer_3)
    input_encoded = Dropout(0.25)(input_encoded)

    embedding = Embedding(nb_words, 150,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=False)

    em_embeddings = Embedding(2, 1,
                        input_length=max_sequence_length,
                        trainable=True)   

    #q1_embed = Concatenate()([embedding(q1), em_embeddings(q1_exact_match)])
    q1_embed = embedding(q1)
    q1_embed = SpatialDropout1D(0.1)(q1_embed)

    #q2_embed = Concatenate()([embedding(q2), em_embeddings(q2_exact_match)])
    q2_embed = embedding(q2)
    q2_embed = SpatialDropout1D(0.1)(q2_embed)

    th = TimeDistributed(Highway(activation='relu'))
    q1_embed = th(q1_embed)
    q2_embed = th(q2_embed)
        
    q1_aligned, q2_aligned = soft_attention_alignment(q1_embed, q2_embed)
    q1_vec = Concatenate()([q1_embed, q2_aligned, substract(q1_embed, q2_aligned), Multiply()([q1_embed, q2_aligned])])
    q2_vec = Concatenate()([q2_embed, q1_aligned, substract(q2_embed, q1_aligned), Multiply()([q2_embed, q1_aligned])])

    dense_compares = [
        Dense(300, activation='elu'),
        Dropout(0.2),
        Dense(200, activation='elu'),
        Dropout(0.2),
    ]

    q1_compared = time_distributed(q1_vec, dense_compares)
    q2_compared = time_distributed(q2_vec, dense_compares)

    q1_rep = apply_multiple(q1_compared, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compared, [GlobalAvgPool1D(), GlobalMaxPool1D()])    

    h_all = Concatenate()([q1_rep, q2_rep])
    h_all = BatchNormalization()(h_all)

    h_all = Dense(256, activation='elu')(h_all)
    h_all = Dropout(0.2)(h_all)
    h_all = BatchNormalization()(h_all)

    h_all = Dense(256, activation='elu')(h_all)
    h_all = Dropout(0.2)(h_all)
    h_all = BatchNormalization()(h_all)    

    out_ = Dense(3, activation='softmax')(h_all)

    model = Model(inputs=[q1, q2, input_layer_3, q1_exact_match, q2_exact_match], outputs=out_)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6, clipnorm=1.5, amsgrad=True), loss='categorical_crossentropy',
    metrics=['accuracy', weighted_accuracy])
    model.summary()
    return model   