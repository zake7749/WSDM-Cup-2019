"""Input and data related data structures.

Largely copied from here:
https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py
"""
import csv
import os
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
    SequentialSampler
from bert import logger


def processor(task_name, model_name, max_seq_len, tokenizer, data_dir,
              batch_size):
    if task_name == 'arct':
        if model_name == 'arct2':
            return ARCT2Processor(data_dir, max_seq_len, tokenizer, batch_size)
        else:
            raise ValueError('Unexpected model_name for arct: %r' % model_name)
    if task_name == 'triples':
        if model_name == 'arct2':
            return ARCT2Processor(data_dir, max_seq_len, tokenizer, batch_size)
        else:
            raise ValueError('Unexpected model_name for triples: %r'
                             % model_name)
    else:
        raise ValueError('Unexpected task name %r' % task_name)


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if isinstance(example.label, int):
            label_id = label_map[example.label] \
                if example.label is not None else None
        elif isinstance(example.label, list):  # pseudo labels
            label_id = example.label  # send list of floats to what is next
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then each
    # token that's truncated likely contains more information than a longer
    # sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For
              single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
              sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        info = 'Guid:    %s\n' % self.guid
        info += 'Text A:  %s\n' % self.text_a
        info += 'Text b:  %s\n' % self.text_b
        info += 'Label:   %s' % self.label
        return info


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, delimiter='\t', quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b,
                             label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b,
                             label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None,
                             label=label))
        return examples


class WSDMProcessor(DataProcessor):

    def get_train_examples(self, data_dir, subset=None):
        data = self._read_tsv(os.path.join(data_dir, 'train.csv'),
            delimiter=',',
            quotechar='"')
        if subset:
            data = data[0:subset]
        return self._create_examples(data, 'train')

    def get_dev_examples(self, data_dir, subset=None):
        data = self._read_tsv(os.path.join(data_dir, 'dev.csv'),
            delimiter=',',
            quotechar='"')
        if subset:
            data = data[0:subset]
        return self._create_examples(data, 'train')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'test.csv'),
                           delimiter=',',
                           quotechar='"'),
            'test')

    def get_labels(self):
        """See base class."""
        return ["agreed", "disagreed", "unrelated"]

    def _create_examples(self, lines, set_type):
        examples = []
        for line in lines[1:]:  # first line is the headers
            guid = line[0]
            text_a = line[3]
            text_b = line[4]
            label = line[7] if set_type == 'train' else None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b,
                             label=label))
        return examples


class WSDMPseudoProcessor(DataProcessor):

    def get_train_examples(self, data_dir, subset=None):
        data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        if subset:
            data = data.iloc[0:subset]
        return self.create_examples(data, 'train')

    def get_dev_examples(self, data_dir, subset=None):
        data = pd.read_csv(os.path.join(data_dir, 'dev.csv'))
        if subset:
            data = data.iloc[0:subset]
        return self.create_examples(data, 'dev')

    def get_test_examples(self, data_dir):
        data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        return self.create_examples(data, 'test')

    def get_labels(self):
        """See base class."""
        return ["agreed", "disagreed", "unrelated"]

    @staticmethod
    def create_examples(data, set_type):
        examples = []
        for i, row in data.iterrows():
            guid = row['id']
            text_a = row['title1_zh']
            text_b = row['title2_zh']
            if not isinstance(text_b, str):
                text_b = ''
            label = [row['agreed'], row['disagreed'], row['unrelated']]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b,
                             label=label))
        return examples


class ARCTProcessor(DataProcessor):

    def get_train_examples(self, data_dir, augmented=False, subset=None):
        if augmented:
            df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
            df_dev = pd.read_csv(os.path.join(data_dir, 'dev.csv'))
            df = pd.concat([df_train, df_dev])
        else:
            df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        if subset:
            df = df.iloc[0:subset]
        return self._create_examples(df)

    def get_dev_examples(self, data_dir, augmented=False, subset=None):
        if augmented:
            df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        else:
            df = pd.read_csv(os.path.join(data_dir, 'dev.csv'))
        if subset:
            df = df.iloc[0:subset]
        return self._create_examples(df)

    def get_test_examples(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        return self._create_examples(df)

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_a = self._append_claim_reason(line['claim'], line['reason'])
            text_b_0 = line['warrant0']
            text_b_1 = line['warrant1']
            label = int(line['label'])
            examples.append([
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b_0,
                             label=label),
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b_1,
                             label=label)])
        return examples

    def _append_claim_reason(self, claim, reason):
        if claim[-1] != '.':
            claim += '.'
        if reason[-1] != '.':
            reason += '.'
        return '%s %s' % (claim, reason)


class ARCT2Processor:

    def __init__(self, data_dir, max_seq_len, tokenizer, batch_size):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def get_train_examples(self, augmented=False, subset=None):
        if augmented:
            df_train = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
            df_dev = pd.read_csv(os.path.join(self.data_dir, 'dev.csv'))
            df = pd.concat([df_train, df_dev])
        else:
            df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        if subset:
            df = df.iloc[0:subset]
        examples, labels = self.create_examples(df)
        dataloader = self.get_dataloader(examples, labels, RandomSampler)
        return len(examples), dataloader

    def get_dev_examples(self, augmented=False, subset=None):
        if augmented:
            df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        else:
            df = pd.read_csv(os.path.join(self.data_dir, 'dev.csv'))
        if subset:
            df = df.iloc[0:subset]
        examples, labels = self.create_examples(df)
        dataloader = self.get_dataloader(examples, labels, SequentialSampler)
        return len(examples), dataloader

    def get_test_examples(self):
        df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        examples, labels = self.create_examples(df)
        dataloader = self.get_dataloader(examples, labels, SequentialSampler)
        return len(examples), dataloader

    def get_labels(self):
        return [0, 1]

    def create_examples(self, df):
        examples = []
        labels = []
        for _, x in df.iterrows():
            guid = x['#id']
            labels.append(int(x['label']))
            c_w0 = InputExample(
                guid=guid,
                text_a=x['claim'],
                text_b=x['warrant0'],
                label=int(x['label']))
            c_w1 = InputExample(
                guid=guid,
                text_a=x['claim'],
                text_b=x['warrant1'],
                label=int(x['label']))
            r_w0 = InputExample(
                guid=guid,
                text_a=x['reason'],
                text_b=x['warrant0'],
                label=int(x['label']))
            r_w1 = InputExample(
                guid=guid,
                text_a=x['reason'],
                text_b=x['warrant1'],
                label=int(x['label']))
            w0_w1 = InputExample(
                guid=guid,
                text_a=x['warrant0'],
                text_b=x['warrant1'],
                label=int(x['label']))
            examples.append([c_w0, c_w1, r_w0, r_w1, w0_w1])
        labels = torch.tensor(labels, dtype=torch.long)
        return examples, labels

    def get_dataloader(self, examples, labels, sampler_type):
        x = [convert_examples_to_features(e, self.get_labels(), self.max_seq_len, self.tokenizer) for e in examples]
        cw0_i = torch.tensor([f[0].input_ids for f in x], dtype=torch.long)
        cw0_m = torch.tensor([f[0].input_mask for f in x], dtype=torch.long)
        cw0_s = torch.tensor([f[0].segment_ids for f in x], dtype=torch.long)
        cw1_i = torch.tensor([f[1].input_ids for f in x], dtype=torch.long)
        cw1_m = torch.tensor([f[1].input_mask for f in x], dtype=torch.long)
        cw1_s = torch.tensor([f[1].segment_ids for f in x], dtype=torch.long)
        rw0_i = torch.tensor([f[2].input_ids for f in x], dtype=torch.long)
        rw0_m = torch.tensor([f[2].input_mask for f in x], dtype=torch.long)
        rw0_s = torch.tensor([f[2].segment_ids for f in x], dtype=torch.long)
        rw1_i = torch.tensor([f[3].input_ids for f in x], dtype=torch.long)
        rw1_m = torch.tensor([f[3].input_mask for f in x], dtype=torch.long)
        rw1_s = torch.tensor([f[3].segment_ids for f in x], dtype=torch.long)
        w0w1_i = torch.tensor([f[4].input_ids for f in x], dtype=torch.long)
        w0w1_m = torch.tensor([f[4].input_mask for f in x], dtype=torch.long)
        w0w1_s = torch.tensor([f[4].segment_ids for f in x], dtype=torch.long)
        data = TensorDataset(cw0_i, cw0_m, cw0_s, cw1_i, cw1_m, cw1_s,
                             rw0_i, rw0_m, rw0_s, rw1_i, rw1_m, rw1_s,
                             w0w1_i, w0w1_m, w0w1_s, labels)
        sampler = sampler_type(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)
