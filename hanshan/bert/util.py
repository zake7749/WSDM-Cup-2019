"""Helper functions.

Largely copied from here:
https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py
"""
import os
import argparse
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from bert import logger
from bert.inputs import InputFeatures


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('run_name',
                        type=str,
                        help='The name of this run (for saving checkpoints).')
    parser.add_argument('model_name',
                        type=str,
                        help='The name of the model to use.')
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv "
                             "files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: "
                             "bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, "
                             "bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='arct',
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model "
                             "checkpoints will be written.")

    # Other parameters
    parser.add_argument('--subset',
                        default=None,
                        type=int,
                        help='Takes a subset of the train data.')
    parser.add_argument('--dev_subset',
                        default=None,
                        type=int,
                        help='Takes a subset of the dev data.')
    parser.add_argument("--max_seq_len",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after "
                             "WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, "
                             "and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear "
                             "learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the "
                             "optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can '
                             'improve fp16 convergence.')

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                                        and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend taking care of syncing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in "
                        "distributed training")
            args.fp16 = False  # https://github.com/pytorch/pytorch/pull/13496
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu,
                bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.batch_size
                                / args.gradient_accumulation_steps)

    if not args.do_train and not args.do_eval:
        raise ValueError("One of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is "
                         "not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    return args, device, n_gpu, task_name


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
            try:
                tokens_b = tokenizer.tokenize(example.text_b)
            except TypeError as e:
                print(example.text_a)
                print(example.text_b)
                print(example.label)
                print(example.guid)
                raise e 

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

        if isinstance(example.label, str):
            label_id = label_map[example.label] if example.label is not None else None
        else:
            label_id = example.label
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


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


class Saver:
    """For loading and saving state dicts.

    The load and save methods accept a model argument. This model is expected to
    be a torch.nn.Module that additionally defines "name" (String) and
    "optimizer" (PyTorch optimizer module) attributes.

    The checkpoint locations are given by the ckpt_dir argument given to the
    constructor, which defines the base directory, and then given the model.name
    attribute saves the files to:
      ckpt_dir/model.name_{model, optim}_{best, latest}
    """

    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir

    def ckpt_path(self, name, module, is_best):
        """Get the file path to a checkpoint.

        Args:
          name: String, the model name.
          module: String in {model, optim}.
          is_best: Bool.

        Returns:
          String.
        """
        return os.path.join(
            self.ckpt_dir,
            '%s_%s_%s' % (name, module, 'best' if is_best else 'latest'))

    def load(self, model, name, is_best=False, load_to_cpu=False,
             load_optimizer=True, replace_model=None, exclude_optim=None,
             ignore_missing=False):
        """Load model and optimizer state dict.

        Args:
          model: a PyTorch model that defines an optimizer attribute.
          is_best: Bool, indicates whether the checkpoint is the best tuning
            accuracy. If not it is "latest".
          load_to_cpu: Bool, for loading GPU trained parameters on cpu. Default
            is False.
          load_optimizer: Bool, defaults to True.
          replace_model: List of strings of sub-modules to exclude from loading.
            For example in transfer we mightn't want to load the old word
            embeddings. The keys in the dict are like `_word_embedding.weight`,
            so to exclude that one pass ['_word_embedding'] as this arg. It will
            also work for attributes of attributes. I.e. ['sub_mod.embeds'] will
            filter out `sub_mod.embeds.weight0`.
          exclude_optim: List of strings of param groups to exclude from
            loading.
          ignore_missing: Bool, if True will ignore parameters in the saved
            state dict that are missing in the model - e.g. as might occur in a
            transfer scenario. Default is False.
        """
        model_path = self.ckpt_path(name, 'model', is_best)
        model_state_dict = self.get_state_dict(model_path, load_to_cpu)
        model_state_dict = self.replace_model_state(
            model_state_dict, replace_model)
        if ignore_missing:
            model_state_dict = self.drop_missing(model, model_state_dict)
        model.load_state_dict(model_state_dict)
        if load_optimizer:
            optim_path = self.ckpt_path(model.name, 'optim', is_best)
            optim_state_dict = self.get_state_dict(optim_path, load_to_cpu)
            model.optimizer.load_state_dict(optim_state_dict)

    @staticmethod
    def drop_missing(model, saved_state_dict):
        return {k: v for k, v in saved_state_dict.items()
                if k in model.state_dict().keys()}

    @staticmethod
    def replace_model_state(state_dict, replace):
        if replace is not None:
            for name, tensor in replace.items():
                state_dict[name] = tensor
        return state_dict

    @staticmethod
    def filter_optim_state_dict(state_dict, exclude):
        if exclude is not None:
            raise NotImplementedError  # TODO
        else:
            return state_dict

    @staticmethod
    def get_state_dict(path, load_to_cpu):
        """Get a state dict from a save file.

        Args:
          path: String.
          load_to_cpu: Bool.
        """
        if not torch.cuda.is_available() or load_to_cpu:
            return torch.load(path, map_location=lambda storage, loc: storage)
        else:
            return torch.load(path)

    def save(self, model, name, is_best, save_optim=False):
        """Save a model and optimizer state dict.

        Args:
          model: a PyTorch model that defines an optimizer attribute.
          is_best: Bool, indicates whether the checkpoint is the best tuning
            accuracy. If not it is "latest".
        """
        model_path = self.ckpt_path(name, 'model', False)
        torch.save(model.state_dict(), model_path)
        if is_best:
            model_path = self.ckpt_path(name, 'model', True)
            torch.save(model.state_dict(), model_path)
        if save_optim:
            optim_path = self.ckpt_path(name, 'optim', is_best)
            torch.save(model.optimizer.state_dict(), optim_path)


class SoftCrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, target):
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num
        return loss
