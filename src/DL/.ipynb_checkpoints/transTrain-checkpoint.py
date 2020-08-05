'''
@Author: your name
@Date: 2020-04-09 18:08:01
@LastEditTime: 2020-04-10 13:36:15
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /textClassification/src/DL/transTrain.py
'''
from __future__ import absolute_import, division, print_function

import glob
import logging
import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (confusion_matrix, accuracy_score, matthews_corrcoef)
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig, RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLNetConfig, XLNetForSequenceClassification,
                          XLNetTokenizer, get_linear_schedule_with_warmup)

from __init__ import *
from src.data.dataset import MyDataset, collate_fn
from src.utils import config
from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = {
    'data_dir': config.root_path + '/data/',
    'model_type': 'bert',
    'model_name': config.root_path + '/model/bert/',
    'output_dir': config.root_path + '/model/bert_classifier',
    'do_train': True,
    'do_eval': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_length': 200,
    'train_batch_size': 32,
    'eval_batch_size': 32,
    'is_cuda': True,
    'gradient_accumulation_steps': 1,
    'num_train_epochs': 20,
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'logging_steps': 50,
    'evaluate_during_training': True,
    'save_steps': 10000,
    'overwrite_output_dir': True
}

device = torch.device("cuda" if args['is_cuda'] else "cpu")

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'roberta':
    (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

# config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]

model_config = config_class.from_pretrained(args['model_name'], num_labels=36)
tokenizer = tokenizer_class.from_pretrained(args['model_name'])
model = model_class.from_pretrained(args['model_name'], config=model_config)
model.to(device)

train_dataset = MyDataset(args['data_dir'] + 'train_clean.tsv',
                          max_length=args['max_length'],
                          tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset,
                              batch_size=args['train_batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)


def train(train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()

    t_total = len(train_dataloader) // args[
        'gradient_accumulation_steps'] * args['num_train_epochs']

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        args['weight_decay']
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0
    }]

    warmup_steps = math.ceil(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args[
        'warmup_steps']

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args['learning_rate'],
                      eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args['warmup_steps'],
        num_training_steps=t_total)

    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args['fp16_opt_level'])

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d",
                args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids':
                batch[0],
                'attention_mask':
                batch[1],
                'token_type_ids':
                batch[2] if args['model_type'] in ['bert', 'xlnet'] else
                None,  # XLM don't use segment_ids
                'labels':
                batch[3]
            }
            outputs = model(**inputs)
            loss = outputs[
                0]  # model outputs are always tuple in pytorch-transformers (see doc)
            print("\r%f" % loss, end='')

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                               args['max_grad_norm'])

            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args['logging_steps'] > 0 and global_step % args[
                        'logging_steps'] == 0:
                    # Log metrics
                    if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value,
                                                 global_step)
                    tb_writer.add_scalar('lr',
                                         scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) /
                                         args['logging_steps'], global_step)
                    logging_loss = tr_loss

                if args['save_steps'] > 0 and global_step % args[
                        'save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args['output_dir'],
                        'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(
                        model, 'module'
                    ) else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, tr_loss / global_step


def get_mismatched(labels, preds):
    mismatched = labels != preds
    examples = pd.read_csv(args['data_dir'] + 'dev_clean.tsv', sep='\t')['label'].values.tolist()
    wrong = [i for (i, v) in zip(examples, mismatched) if v]

    return wrong


def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    result = confusion_matrix(labels, preds).ravel()
    if len(result) == 4:
        tn, fp, fn, tp = result
    else:
        tn, fp, fn, tp = 0 ,0 , 0, 0
    return {
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }, get_mismatched(labels, preds)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)


def evaluate(model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args['output_dir']

    results = {}

    dev_dataset = MyDataset(args['data_dir'] + 'dev_clean.tsv',
                          max_length=args['max_length'],
                          tokenizer=tokenizer)
    dev_dataloader = DataLoader(dev_dataset,
                              batch_size=args['eval_batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dev_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(dev_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids':
                batch[0],
                'attention_mask':
                batch[1],
                'token_type_ids':
                batch[2] if args['model_type'] in ['bert', 'xlnet'] else
                None,  # XLM don't use segment_ids
                'labels':
                batch[3]
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      inputs['labels'].detach().cpu().numpy(),
                                      axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result, wrong = compute_metrics(preds, out_label_ids)
    results.update(result)
    acc = accuracy_score(out_label_ids, preds)
    logger.info("***** Eval ACC {} *****".format(acc))

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results, wrong


if args['do_train']:
    global_step, tr_loss = train(train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])
    logger.info("Saving model checkpoint to %s", args['output_dir'])

    model_to_save = model.module if hasattr(
        model,
        'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args['output_dir'])
    tokenizer.save_pretrained(args['output_dir'])
    torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))

results = {}
if args['do_eval']:
    checkpoints = [args['output_dir']]
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        model = model_class.from_pretrained(checkpoint)
        model.to(device)
        result, wrong_preds = evaluate(model, tokenizer, prefix=global_step)
        result = dict(
            (k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)
