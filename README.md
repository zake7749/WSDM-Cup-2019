# Fake News Classification

> This is the code to reproduce our results in WSDM 2019. Since we were in a race against the clock, this project is kind of dirty currently. I will upload the refactored version on another branch as soon as possible.

## Documents

* [Slides](https://docs.google.com/presentation/d/1RFKX6dJT4-MdwvKA2gglTXy1VJumkVDHU-USJpGDllc/edit?usp=sharing)
* [Paper](https://people.eng.unimelb.edu.au/jianzhongq/wsdm19-cup-reports/reports/report3.pdf)

## Reproduce our results

### 1. Setup

1. Clone this project.

2. Download the datasets and extract them under the folder `zake7749/data/dataset`

```
|-- dataset
    |-- sample_submission.csv
    |-- test.csv
    `-- train.csv
```

3. Prepare the embedding models

We use 2 open-source pretrained word embeddings in this competiton:

* [Tencent AI Lab Embedding Corpus for Chinese Words and Phrases](https://ai.tencent.com/ailab/nlp/embedding.html)
* [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
    * We select the [SGNS version](https://pan.baidu.com/s/1oJol-GaRMk4-8Ejpzxo6Gw) on word and n-gram level, trained with the mixed-large corpus

And put these two embeddings under the folder `zake7749/data/wordvec/`

```
|-- wordvec
    |-- Tencent_AILab_ChineseEmbedding.txt
    `-- sgns.merge.bigram
```

### 2. Instructions

The notebooks are under the folder `zake7749/code`

#### Pre-processing

1. Execute `Stage 1.1. Preprocessing-on-word-level.ipynb`
2. Execute `Stage 1.2. Preprocessing-on-char-level.ipynb`

These notebooks would generate 8 cleaned datasets under `zake7749/data/processed_dataset`. 

```
.
|-- engineered_chars_test.csv
|-- engineered_chars_train.csv
|-- engineered_words_test.csv
|-- engineered_words_train.csv
|-- processed_chars_test.csv
|-- processed_chars_train.csv
|-- processed_words_test.csv
`-- processed_words_train.csv

```

#### Train the char-level embedding

Execute `Stage 1.3. Train-char-embeddings`, which would output 3 char embeddings under `zake7749/data/wordvec/`

```
|-- wordvec
    |-- Tencent_AILab_ChineseEmbedding.txt
    |-- fasttext-50-win3.vec
    |-- sgns.merge.bigram
    |-- zh-wordvec-50-cbow-windowsize50.vec
    `-- zh-wordvec-50-skipgram-windowsize7.vec
```

#### Train the base models (LB 0.84 ~ 0.86)

* Execute `Stage 2. First-Level-with-char-level.ipynb`
* Execute `Stage 2. First-Level-with-word-level.ipynb`

#### Ensemble the predictions of base models (LB 0.873)

1. Execute `Stage 3.1. First-level-ensemble-ridge-regression`
2. Execute `Stage 3.2. First-level-ensemble-with-LGBM-each-side`
3. Execute `Stage 3.3. First-level-ensemble-with-LGBM`
4. Execute `Stage 3.4. First-level-ensemble-with-NN`
5. Execute `Stage 3.5. Second-level-ensemble`

#### Fine-tune the cls vector of BERT (LB 0.867)

* Run script `hanshan/bert/train_wsdm.sh`
* To get predictions file to submit at this stage run `zake7749/bert/data/probs_to_preds.py`

#### Blend the predictions of ensemble NNs with BERT (LB 0.874)

* Execute `Stage 3.6. Bagging-with-BERT`

** Note: Please change the path of sec_stacking_df to the corresponding file **

#### Fine-tune the base models with noisy labels (LB 0.86 ~ 0.875)

* Execute `Stage 4.1. Fine-tune-word-levels.ipynb`
* Execute `Stage 4.2. Fine-tune-word-levels-ScaleValid.ipynb`
* Execute `Stage 4.3. Fine-tune-char-levels.ipynb`
* Execute `Stage 4.4. Fine-tune-char-levels-ScaleValid.ipynb`

#### Fine-tune the cls vector of BERT with noisy labels (LB 0.880)

* Run `hanshan/prep_pseudo_labels.py`
* Run script `hanshan/bert/train_wsdm_pl.sh`

#### Ensemble the predictions of fine-tuned base models (LB 0.879)

1. Execute `Stage 5.1. First-level-fine-tuned-ensemble-ridge-regression.ipynb`
2. Execute `Stage 5.2. First-level-fine-tuned-ensemble-withNN.ipynb`
3. Execute `Stage 5.3. First-level-fine-tuned-ensemble-with-LGBM.ipynb`
4. Execute `Stage 5.4. Second-level-fine-tuned-ensemble.ipynb`

#### Final Blending with post-processing (LB 0.881)

1. Execute `Stage 9. High-Ground.ipynb`
2. Execute `Stage 42. Final Answer.ipynb`

The final prediction `final_answer.csv` would be generated under the folder `zake7749/data/high_ground/`
