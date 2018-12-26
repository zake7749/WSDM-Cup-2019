USE_CUDA = True
MODEL_CHECKPOINT_FOLDER = "checkpoints/"
TEMPORARY_CHECKPOINTS_PATH = "temporary_checkpoints/"

MAX_SENTENCE_LENGTH = 56
MAX_GRADIENT = 10
MAX_CHAR_LENGTH = 11


META_FEATURES = ['bm25_q1_to_q2', 'bm25_q2_to_q1', 'weighted_cosine_sim',
       'len_word_max', 'len_word_min', 'len_char_max', 'len_char_min',
       'word_length_diff', 'char_length_diff', 'len_diff_remove_stopwords',
       'word_match', 'tfidf_word_match', 'shared_count', 'bigram_corr', 'trigram_corr',
       'word_match_no_stopwords', 'unique_word_ratio', 'cosine_sim',
       'manhattan_dis', 'eucledian_dis', 'jaccard_dis', 'minkowsk_dis',
       'fuzzy_ratio', 'fuzzy_set_ratio', 'fuzzy_partial_ratio',
       'fuzzy_token_sort_ratio', 'fuzzy_qratio', 'fuzzy_WRatio',
       'longest_substr_ratio', 'cómo_both', 'simhash_distance', 'simhash_distance_2gram',
       'simhash_distance_3gram', 'simhash_distance_ch_2gram',
       'simhash_distance_ch_3gram',
]