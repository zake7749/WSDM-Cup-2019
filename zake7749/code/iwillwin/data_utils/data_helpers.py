import pandas as pd
import numpy as np
import re
import gensim
import thulac
import jieba

from collections import Counter
from gensim import corpora
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

from iwillwin.config import dataset_config, model_config
from iwillwin.data_utils.tokenizer import StableTokenizer
from iwillwin.data_utils.feature_engineering import FeatureCreator, CharFeatureCreator

class DataLoader(object):

    def __init__(self):
        pass

    def load_dataset(self, dataset_path, names):
        '''return a pandas processed csv'''
        return pd.read_csv(dataset_path)

    def load_embedding(self, embedding_path, keras_like=True):
        '''return a dict whose key is word, value is pretrained word embedding'''
        if keras_like:
            embeddings_index = {}
            f = open(embedding_path, 'r', encoding='utf-8')
            for line in f:
                values = line.split()
                try:
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except:
                    print("Err on ", values[:2])
            f.close()
            print('Total %s word vectors.' % len(embeddings_index))
            return embeddings_index

class DataTransformer(object):
    
    def __init__(self, max_num_words, max_sequence_length, char_level, normalization, features_processed=False):
        self.data_loader = DataLoader()
        self.train_df = self.data_loader.load_dataset(dataset_config.DATASET_TRAIN_PATH, names=None)
        self.test_df = self.data_loader.load_dataset(dataset_config.DATASET_TEST_PATH, names=None)
        self.train_df = self.train_df.drop_duplicates()

        self.processed_train_set = dataset_config.PROCESSED_WORDS_TRAIN_SET
        self.processed_test_set = dataset_config.PROCESSED_WORDS_TEST_SET
        self.engineered_train_set = dataset_config.ENGINEERED_WORDS_TRAIN_SET
        self.engineered_test_set = dataset_config.ENGINEERED_CHARS_TEST_SET

        self.language_colunm_1 = 'spn_1'
        self.language_colunm_2 = 'spn_2'

        for df in [self.train_df, self.test_df,]:
            df['raw_spn_1'] = df['title1_zh'].values
            df['raw_spn_2'] = df['title2_zh'].values

        self.max_num_words = max_num_words
        self.max_sequence_length = max_sequence_length
        self.max_char_length = model_config.MAX_CHAR_LENGTH
        self.char_level = char_level
        self.tokenizer = None
        self.char_tokenizer = None
        self.normalization = normalization
        
        self.features_processed = features_processed
        if not self.features_processed:
            self.feature_creator = FeatureCreator(self.train_df, self.test_df, self.data_loader, normalization=normalization)

    def expand_features(self):
        self.train_df, self.test_df = self.feature_creator.create_features()
        self.train_df.to_csv(self.processed_train_set, index=False, encoding='utf-8')
        self.test_df.to_csv(self.processed_test_set, index=False, encoding='utf-8')
        return self.train_df, self.test_df

    def apply_normalization(self, train_df, test_df):
        all_df = pd.concat((train_df, test_df))
        for column in model_config.META_FEATURES:
            if column in all_df.columns:
                scaler = MinMaxScaler()
                all_df[column] = scaler.fit_transform(all_df[column].values.reshape(-1, 1))
            else:
                print("[DH-Norm] The column", column, "is not in the dataframe.")
        train_df, test_df = all_df.iloc[:len(train_df)], all_df.iloc[len(train_df):]
        return train_df, test_df

    def prepare_data(self, drop_stopwords=False, dual=False):
        if not self.features_processed:
            print("[DataHelper Error] Please run the notebook Preprocessing.ipynb before calling prepare_data.")
            exit()
            #self.train_df, self.test_df = self.expand_features()
        else:
            self.train_df = pd.read_csv(self.engineered_train_set, encoding='utf-8')
            self.test_df = pd.read_csv(self.engineered_test_set, encoding='utf-8')

            for df in [self.train_df, self.test_df]:
                df['spn_1'] = df['title1_zh'].apply(lambda v: self.preprocessing(v))
                df['spn_2'] = df['title2_zh'].apply(lambda v: self.preprocessing(v))

        self.train_df = self.train_df.fillna(0)
        self.test_df = self.test_df.fillna(0)

        if self.normalization:
            print("[DataHelper] Apply normalization on value-type columns")
            self.train_df, self.test_df = self.apply_normalization(self.train_df, self.test_df)

        train_sentences_1 = self.train_df[self.language_colunm_1].fillna("no comment").values
        train_sentences_2 = self.train_df[self.language_colunm_2].fillna("no comment").values

        test_sentences_1 = self.test_df[self.language_colunm_1].fillna("no comment").values
        test_sentences_2 = self.test_df[self.language_colunm_2].fillna("no comment").values

        training_labels = self.train_df["label"].values
        label2id = {'unrelated':0, 'agreed':1, 'disagreed':2}
        training_labels = np.array([label2id[label] for label in training_labels])

        print("Doing preprocessing...")
        self.train_sentences_1 = [self.preprocessing(text, drop_stopwords) for text in train_sentences_1]
        self.train_sentences_2 = [self.preprocessing(text, drop_stopwords) for text in train_sentences_2]

        self.test_sentences_1 = [self.preprocessing(text, drop_stopwords) for text in test_sentences_1]
        self.test_sentences_2 = [self.preprocessing(text, drop_stopwords) for text in test_sentences_2]

        print("Transforming words to indices...")
        self.build_tokenizer(self.train_sentences_1 + self.train_sentences_2 + self.test_sentences_1 + self.test_sentences_2) # keep the smae order
     
        # prepare training pairs
        train_sentence_1 = self.get_padded_sequences(self.train_sentences_1, maxlen=self.max_sequence_length)
        train_sentence_2 = self.get_padded_sequences(self.train_sentences_2, maxlen=self.max_sequence_length)

        # prepare testing pairs
        test_sentence_1 = self.get_padded_sequences(self.test_sentences_1, maxlen=self.max_sequence_length)
        test_sentence_2 = self.get_padded_sequences(self.test_sentences_2, maxlen=self.max_sequence_length)    

        ## meta-features
        train_features = self.train_df[model_config.META_FEATURES].values
        test_features = self.test_df[model_config.META_FEATURES].values

        print('Shape of data tensor:', train_sentence_1.shape, train_sentence_2.shape)
        print('Shape of label tensor:', training_labels.shape)

        print("Preprocessed.")
        return (train_sentence_1, train_sentence_2, train_features), (test_sentence_1, test_sentence_2, test_features), training_labels

    def get_padded_sequences(self, sentences, maxlen):
        sentences = self.tokenizer.texts_to_sequences(sentences)
        sentences = pad_sequences(sentences, maxlen=maxlen)
        return sentences

    def preprocessing(self, text, drop_stopwords=False):
        if type(text) == float:
            return 'e'        
        
        words = [w for w in jieba.cut(text)]
        text = " ".join(words)    
        text = re.sub(r"\<i\>", "", text)
        text = re.sub(r"|", "", text)
        text = re.sub(r";", "", text)
        text = re.sub(r"，", "", text)
        text = re.sub(r"！ ", "", text)
        text = re.sub(r"!", "", text)
        text = re.sub(r"¿", "", text)
        text = re.sub(r",", "", text)
        text = re.sub(r"–", "", text)
        text = re.sub(r"−", "", text)
        text = re.sub(r"\.", "", text)
        text = re.sub(r"!", "", text)
        text = re.sub(r"\/", "", text)
        text = re.sub(r"_", "", text)
        text = re.sub(r"\?", "", text)
        text = re.sub(r"？", "", text)
        text = re.sub(r"\^", "", text)
        text = re.sub(r"\+", "", text)
        text = re.sub(r"\-", "", text)
        text = re.sub(r"\=", "", text)
        text = re.sub(r"#", "", text)

        text = re.sub(r"'", "", text)
        return text

    def build_embedding_matrix(self, embeddings_index):
        nb_words = min(self.max_num_words, len(embeddings_index))
        embedding_matrix = np.zeros((nb_words, 300))
        word_index = self.tokenizer.word_index
        null_words = open('null-word.txt', 'w', encoding='utf-8')

        for word, i in word_index.items():

            if i >= self.max_num_words:
                null_words.write(word + '\n')
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                null_words.write(word + '\n')
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        return embedding_matrix

    def build_tokenizer(self, comments):
        self.tokenizer = StableTokenizer(num_words=self.max_num_words, char_level=self.char_level)
        self.tokenizer.fit_on_texts(comments)


class CharDataTransformer(DataTransformer):

    def __init__(self, max_num_words, max_sequence_length, char_level, normalization, features_processed=False):
        super(CharDataTransformer, self).__init__(max_num_words, max_sequence_length, char_level, normalization, features_processed)

        self.processed_train_set = dataset_config.PROCESSED_CHARS_TRAIN_SET
        self.processed_test_set = dataset_config.PROCESSED_CHARS_TEST_SET
        self.engineered_train_set = dataset_config.ENGINEERED_CHARS_TRAIN_SET
        self.engineered_test_set = dataset_config.ENGINEERED_CHARS_TEST_SET

        if not self.features_processed:
            self.feature_creator = CharFeatureCreator(self.train_df, self.test_df, self.data_loader, normalization=normalization)

    def preprocessing(self, text, drop_stopwords=False):
        if type(text) == float:
            return 'e'        
        text = re.sub(r";", "", text)
        text = re.sub(r"<i>", "", text)
        
        text = re.sub(r"’", "”", text)
        text = re.sub(r"‘", "“", text)

        text = re.sub(r"!", "!", text)
        text = re.sub(r"！", "!", text)
        text = re.sub(r",", "，", text)
        text = re.sub(r"–", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"_", " ", text)
        text = re.sub(r"\?", "?", text)
        text = re.sub(r"？", "?", text)
        text = re.sub(r"\^", "^", text)
        text = re.sub(r"\+", "+", text)
        text = re.sub(r"\-", "-", text)
        text = re.sub(r"\=", "=", text)
        text = re.sub(r"#", "#", text)

        return text

    def prepare_data(self, drop_stopwords=False, dual=False):
        if not self.features_processed:
            print("[DataHelper Error] Please run the notebook Preprocessing.ipynb before calling prepare_data.")
            exit()
            #self.train_df, self.test_df = self.expand_features()
        else:
            self.train_df = pd.read_csv(self.engineered_train_set, encoding='utf-8')
            self.test_df = pd.read_csv(self.engineered_test_set, encoding='utf-8')

        self.train_df = self.train_df.fillna(0)
        self.test_df = self.test_df.fillna(0)

        if self.normalization:
            print("[DataHelper] Apply normalization on value-type columns")
            self.train_df, self.test_df = self.apply_normalization(self.train_df, self.test_df)

        train_sentences_1 = self.train_df[self.language_colunm_1].fillna("no comment").values
        train_sentences_2 = self.train_df[self.language_colunm_2].fillna("no comment").values

        test_sentences_1 = self.test_df[self.language_colunm_1].fillna("no comment").values
        test_sentences_2 = self.test_df[self.language_colunm_2].fillna("no comment").values

        training_labels = self.train_df["label"].values
        label2id = {'unrelated':0, 'agreed':1, 'disagreed':2}
        training_labels = np.array([label2id[label] for label in training_labels])

        print("Doing preprocessing...")
        self.train_sentences_1 = [self.preprocessing(text, drop_stopwords) for text in train_sentences_1]
        self.train_sentences_2 = [self.preprocessing(text, drop_stopwords) for text in train_sentences_2]

        self.test_sentences_1 = [self.preprocessing(text, drop_stopwords) for text in test_sentences_1]
        self.test_sentences_2 = [self.preprocessing(text, drop_stopwords) for text in test_sentences_2]

        print("Transforming words to indices...")
        self.build_tokenizer(self.train_sentences_1 + self.train_sentences_2 + self.test_sentences_1 + self.test_sentences_2) # keep the smae order
     
        # prepare training pairs
        train_sentence_1 = self.get_padded_sequences(self.train_sentences_1, maxlen=self.max_sequence_length)
        train_sentence_2 = self.get_padded_sequences(self.train_sentences_2, maxlen=self.max_sequence_length)

        # prepare testing pairs
        test_sentence_1 = self.get_padded_sequences(self.test_sentences_1, maxlen=self.max_sequence_length)
        test_sentence_2 = self.get_padded_sequences(self.test_sentences_2, maxlen=self.max_sequence_length)    

        ## meta-features
        train_features = self.train_df[model_config.META_FEATURES].values
        test_features = self.test_df[model_config.META_FEATURES].values

        print('Shape of data tensor:', train_sentence_1.shape, train_sentence_2.shape)
        print('Shape of label tensor:', training_labels.shape)

        print("Preprocessed.")
        return (train_sentence_1, train_sentence_2, train_features), (test_sentence_1, test_sentence_2, test_features), training_labels       

    def build_tokenizer(self, comments):
        self.tokenizer = StableTokenizer(num_words=self.max_num_words, char_level=self.char_level, filters='"$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n')
        self.tokenizer.fit_on_texts(comments)