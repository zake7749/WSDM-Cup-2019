import os

DATASET_ROOT = "../data/dataset/"
DATASET_TRAIN_PATH = os.path.join(DATASET_ROOT, "train.csv")
DATASET_TEST_PATH = os.path.join(DATASET_ROOT, "test.csv")

PROCESSED_DATASET_ROOT = "../data/processed_dataset/"
PROCESSED_WORDS_TRAIN_SET = PROCESSED_DATASET_ROOT + "processed_words_train.csv"
PROCESSED_WORDS_TEST_SET = PROCESSED_DATASET_ROOT + "processed_words_test.csv"
PROCESSED_CHARS_TRAIN_SET = PROCESSED_DATASET_ROOT + "processed_chars_train.csv"
PROCESSED_CHARS_TEST_SET = PROCESSED_DATASET_ROOT + "processed_chars_test.csv"
PROCESSED_TRAIN_SET_DROPS_STOPS = PROCESSED_DATASET_ROOT + "processed_train_drops_stopwords.csv"
PROCESSED_TEST_SET_DROPS_STOPS = PROCESSED_DATASET_ROOT + "processed_test_drops_stopwords.csv"
PROCESSED_TRAIN_SET_DROPS_SHARES = PROCESSED_DATASET_ROOT + "processed_train_drops_shares.csv"
PROCESSED_TEST_SET_DROPS_SHARES = PROCESSED_DATASET_ROOT + "processed_test_drops_shares.csv"

AUGMENTED_TRAIN_SET = DATASET_ROOT + "both_augmented.csv"

ENGINEERED_WORDS_TRAIN_SET = PROCESSED_DATASET_ROOT + "engineered_words_train.csv"
ENGINEERED_WORDS_TEST_SET = PROCESSED_DATASET_ROOT + "engineered_words_test.csv"
ENGINEERED_CHARS_TRAIN_SET = PROCESSED_DATASET_ROOT + "engineered_chars_train.csv"
ENGINEERED_CHARS_TEST_SET = PROCESSED_DATASET_ROOT + "engineered_chars_test.csv"

ENGINEERED_TRAIN_SET_DROPS_STOPS = PROCESSED_DATASET_ROOT + "engineered_train_drops_stopwords.csv"
ENGINEERED_TEST_SET_DROPS_STOPS = PROCESSED_DATASET_ROOT + "engineered_test_drops_stopwords.csv"
ENGINEERED_TRAIN_SET_DROPS_SHARES = PROCESSED_DATASET_ROOT + "engineered_train_drops_shares.csv"
ENGINEERED_TEST_SET_DROPS_SHARES = PROCESSED_DATASET_ROOT + "engineered_test_drops_shares.csv"

WORDVEC_ROOT = "../data/wordvec/"
ENGLISH_WORDVEC_PATH = os.path.join(WORDVEC_ROOT, "wiki.es.vec")
SPANISH_WORDVEC_PATH = os.path.join(WORDVEC_ROOT, "wiki.en.vec")

ENGLISH_EMBEDDING_SIZE = 300