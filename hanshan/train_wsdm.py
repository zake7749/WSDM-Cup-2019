"""Training script virtually copied from here:
https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py
"""
import os
import argparse
import torch
import random
import pickle
import numpy as np
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
    SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from bert import logger, wsdm
from bert.inputs import MrpcProcessor, MnliProcessor, ColaProcessor, \
    WSDMProcessor, ARCTProcessor
from bert.util import convert_examples_to_features, accuracy, \
    copy_optimizer_params_to_model, set_optimizer_params_grad, Saver
from pytorch_pretrained_bert import BertTokenizer, BertAdam, \
    BertForSequenceClassification, PYTORCH_PRETRAINED_BERT_CACHE


def tune(model, eval_dataloader, device, saver, run_name, tune_accs):
    # Tune on the dev set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    with tqdm(total=len(eval_dataloader)) as pbar:
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                loss, logits = model(
                    input_ids, segment_ids, input_mask, label_ids)
            nb_eval_examples += label_ids.size(0)
            logits = logits.detach().cpu().numpy()
            tmp_eval_accuracy = accuracy(logits,
                                         label_ids.detach().cpu().numpy())
            eval_loss += loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            pbar.update()
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    tune_accs.append(eval_accuracy)
    result = {'eval_loss': eval_loss / nb_eval_steps,
              'eval_accuracy': eval_accuracy}

    print('Dev results')
    print(result)
    is_best = eval_accuracy == np.max(tune_accs)
    print('Is best: %s' % is_best)

    saver.save(model, run_name, is_best)

    model.train()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('run_name',
                        type=str,
                        help='The name of this run (for saving checkpoints).')
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
                        default=None,
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
    parser.add_argument('--resume',
                        default=False,
                        action='store_true',
                        help='Whether to resume training.')
    parser.add_argument('--resume_epoch',
                        default=1,
                        type=int,
                        help='The epoch from which to resume.')
    parser.add_argument('--subset',
                        default=None,
                        type=int,
                        help='Takes a subset of the train data.')
    parser.add_argument('--dev_subset',
                        default=None,
                        type=int,
                        help='Takes a subset of the dev data.')
    parser.add_argument("--max_seq_length",
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
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
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

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        'wsdm': WSDMProcessor,
        'arct': ARCTProcessor,
    }

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

    args.train_batch_size = int(args.train_batch_size
                                / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("One of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is "
                         "not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, args.subset)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size
            / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model,
        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
                  'distributed_{}'.format(args.local_rank),
        num_labels=len(label_list))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # For resuming training
    saver = Saver(ckpt_dir=args.output_dir)
    if args.resume:
        saver.load(model, args.run_name, False, load_optimizer=False)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [
            (n, param.clone().detach().to('cpu').float().requires_grad_()) \
            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [
            (n, param.clone().detach().to('cpu').requires_grad_()) \
            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    #
    # TRAIN

    # Load all data first

    # train
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)
    if args.subset:
        train_features = train_features[0:args.subset]
    all_input_ids = torch.tensor([f.input_ids for f in train_features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features],
                                   dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features],
                                 dtype=torch.long)
    train_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # dev
    #eval_examples = processor.get_dev_examples(
    #    args.data_dir, args.dev_subset)
    #if args.dev_subset:
    #    eval_examples = eval_examples[0:args.dev_subset]
    #eval_features = convert_examples_to_features(
    #    eval_examples, label_list, args.max_seq_length, tokenizer)
    #all_input_ids = torch.tensor(
    #    [f.input_ids for f in eval_features],
    #    dtype=torch.long)
    #all_input_mask = torch.tensor(
    #    [f.input_mask for f in eval_features],
    #    dtype=torch.long)
    #all_segment_ids = torch.tensor(
    #    [f.segment_ids for f in eval_features],
    #    dtype=torch.long)
    #all_label_ids = torch.tensor(
    #    [f.label_id for f in eval_features],
    #    dtype=torch.long)
    #eval_data = TensorDataset(
    #    all_input_ids, all_input_mask, all_segment_ids,
    #    all_label_ids)
    #eval_sampler = SequentialSampler(eval_data)
    #eval_dataloader = DataLoader(
    #    eval_data, sampler=eval_sampler,
    #    batch_size=args.eval_batch_size)

    # test
    test_examples = processor.get_test_examples(args.data_dir)
    ids = [x.guid for x in test_examples]
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in test_features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in test_features],
        dtype=torch.long)
    #all_label_ids = torch.tensor([f.label_id for f in test_features],
    #                             dtype=torch.long)
    test_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    # training process
    global_step = 0 if not args.resume else args.resume_epoch

    if args.do_train:
        # Epochs
        model.train()
        train_accs = []
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss, train_accuracy = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader,
                                              desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss, logits = model(
                    input_ids, segment_ids, input_mask, label_ids)
                logits = logits.detach().cpu().numpy()
                tmp_train_accuracy = accuracy(logits,
                                              label_ids.detach().cpu().numpy())
                train_accuracy += tmp_train_accuracy
                train_accs.append(train_accuracy)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / \
                                                      args.loss_scale
                        is_nan = set_optimizer_params_grad(
                            param_optimizer,
                            model.named_parameters(),
                            test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients,"
                                        "reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(),
                                                       param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

    saver.save(model, 'PseudoFirstLevel', is_best=True)

    #
    # TEST

    if args.do_test:
        model.eval()
        preds = []
        label_dict = dict(zip(label_list, range(len(label_list))))
        rev_label_dict = {i: l for l, i in label_dict.items()}
        with tqdm(total=len(test_dataloader)) as pbar:
            for batch in test_dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids = batch
                with torch.no_grad():
                    logits = model(
                        input_ids, segment_ids, input_mask, None)
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                probs.shape = (logits.size(0), 3)
                preds.append(probs)
                pbar.update()

        output_test_file = '../zake7749/data/bert/bert.csv'
        preds = np.concatenate(preds, axis=0)
        df = pd.DataFrame(preds, columns=processor.get_labels())
        print(df.head())
        df.to_csv(output_test_file, index=False)
        logger.info('Test completed.')


if __name__ == "__main__":
    main()

