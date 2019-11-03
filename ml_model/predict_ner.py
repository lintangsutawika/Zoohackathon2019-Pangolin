# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import json
import random
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils_ner import convert_examples_to_features, read_examples_from_file, get_labels

from transformers import AdamW, WarmupLinearSchedule
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)),
    ())

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer)
}

## Required parameters
data_dir = '/home/nggih/Konvergen/transformers/examples/'
model_type = "roberta"
model_name_or_path = "roberta-base"
output_dir = '/home/nggih/zoohackathon/data/ready/output/' 
labels_path = '/home/nggih/zoohackathon/label/label.txt'
config_name = ''
tokenizer_name = ''
cache_dir = ''
do_lower_case = False
max_seq_length = 256
per_gpu_train_batch_size = 8
per_gpu_eval_batch_size = 8
gradient_accumulation_steps = 1
learning_rate = 2e-5 
weight_decay = 0.0
adam_epsilon = 1e-8
max_grad_norm = 1.0
num_train_epochs = 30
max_steps = -1
warmup_steps = 0
logging_steps = 50
save_steps = 250
eval_all_checkpoints = True
no_cuda = False
seed = 42
local_rank = -1
overwrite_cache = False
n_gpu = 1
# CUDA
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")

def set_seed():
    seeder = 42
    random.seed(seeder)
    np.random.seed(seeder)
    if n_gpu >0:
        torch.cuda.manual_seed_all(seeder)

set_seed()

def evaluate(model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples( tokenizer, labels, pad_token_label_id, mode=mode)

    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if model_type in ["bert", "xlnet"] else None,
                      # XLM and RoBERTa don"t use segment_ids
                      "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    print('length:')
    print(len(out_label_list))
    print(len(preds_list))
    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list)
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def load_and_cache_examples(tokenizer, labels, pad_token_label_id, mode):
    # if local_rank not in [-1, 0] and not evaluate:
        # torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # # Load data features from cache or dataset file
    # cached_features_file = os.path.join(data_dir, "cached_{}_{}_{}".format(mode,
        # list(filter(None, model_name_or_path.split("/"))).pop(),
        # str(max_seq_length)))
    # if os.path.exists(cached_features_file):
        # logger.info("Loading features from cached file %s", cached_features_file)
        # features = torch.load(cached_features_file)
    # else:
    logger.info("Creating features from dataset file at %s", data_dir)
    examples = read_examples_from_file(data_dir, mode)
    print("GOES TO LOAD AND CACHE")
    features = convert_examples_to_features(examples, labels, max_seq_length, tokenizer,
                                            cls_token_at_end=bool(model_type in ["xlnet"]),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ["roberta"]),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(model_type in ["xlnet"]),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
                                            pad_token_label_id=pad_token_label_id
                                            )
    # if local_rank in [-1, 0]:
        # logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save(features, cached_features_file)

    # if local_rank == 0 and not evaluate:
        # torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def categorizing(dt):
    result = dict()
    tag = ''
    temp = ''
    container = []
    print('Categorizing:')
    for i, r in enumerate(dt):
        if tag == r[1]:
            temp += ' '+r[0]
        else:
            container.append((temp, tag))
            tag = r[1]
            temp = r[0]
        if i == len(dt)-1:
            result[tag]=temp
    result = defaultdict(list)
    for temp, tag in container:
        result[tag].append(temp)

    if '' in result:
        for k in ['','NONE']:
            del result[k]
    else:
        for k in ['NONE']:
            del result[k]
        
    return result

def print_result_precision(result):
    for key in sorted(result.keys()):
        print('##############')
        print(f"{key} = {str(result[key])}")

    
def char_indexing(article):
    # {"id": 1, "text": "EU rejects ...", "labels": [[0,2,"ORG"], [11,17, "MISC"], [34,41,"ORG"]]}
    # {"id": 2, "text": "Peter Blackburn", "labels": [[0, 15, "PERSON"]]}
    # {"id": 3, "text": "President Obama", "labels": [[10, 15, "PERSON"]]}
    flag_non_space_string_started = False
    positions = []
    for i, letter in enumerate(article):
        if letter is not ' ':
            if not flag_non_space_string_started:
                positions.append(i)
                flag_non_space_string_started = True
        else:
            flag_non_space_string_started = False
    end_positions = [i-1 for i in positions[1:]]
    end_positions.append(len(article))
    
    start_end = []
    for start, end in zip(positions, end_positions):
        word = article[start:end]
        start_end.append([start, end, word])

    # print(positions)
    return start_end 


def article_to_txt(article):
    article = article.replace('<EOL>', ' ')
    article = article.split(' ')
    article = [i+'\n' for i in article] 

    return article

def get_label_docanno(start_end, container):
    labels = []
    for pos, pred in zip(start_end, container):
        if pos[2] == pred[1]:
            if pred[2] == 'NONE':
                pass
            else:
                labels.append([pos[0], pos[1], pred[2]])
    return labels

def to_docanno(article, labels):
    return {"text": article, "labels": labels}

def run(article):
    original_article = article
    start_end = char_indexing(article)

    article = article_to_txt(article)

    # Prepare CONLL-2003 task
    labels = get_labels(labels_path)
    
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(config_name if config_name else model_name_or_path,
                                          num_labels=num_labels)

    # Predict
    tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=do_lower_case)
    model = model_class.from_pretrained(output_dir)
    model.to(device)
    result, predictions = evaluate( model, tokenizer, labels, pad_token_label_id, mode="test")

    # # Save results
    # print_result_precision(result)

    # Save predictions
    example_id = 0
    prediction_lines = []
    container = []

    # for i, line in enumerate(article):
        # if line.startswith("-DOCSTART-") or line == "" or line == "\n":
            # if not predictions[example_id]:
                # example_id += 1
        # elif predictions[example_id]:
            # word = line.split()[0]
            # pred = predictions[example_id].pop(0)
            # print(start_end[i], word, pred)
            # output_line = str(example_id) + " " +  word + " " + pred + "\n"
            # prediction_lines.append((word, pred))
        # else:
            # logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    for i, line in enumerate(article):
        if line.startswith("-DOCSTART-") or line == "" or line == "\n": #or line.endswith('.') or line.endswith(',') or line.endswith(';') :
            if not predictions[example_id]:
                example_id += 1
                container.append([example_id, line, 'NONE'])
        elif predictions[example_id]:
            word = line.split()[0]
            pred = predictions[example_id].pop(0)
            
            container.append([example_id, word, pred]) 
            prediction_lines.append((word, pred))
        else:
            logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])
    result = categorizing(prediction_lines)
    
    print(len(container))
    print(len(start_end))
    labels = get_label_docanno(start_end, container)
    docanno = to_docanno(original_article, labels)

    print('PREDICTION RESULT:')
    for k,v in result.items():
        print(k,':',v)
        print("###")

    # returns docanno.json 
    print(docanno)
    json_filename = 'docanno.json'
    with open(json_filename, 'w') as f:
        json.dump(docanno, f)
    

    return docanno


print('LOAD predict_ner')
if __name__ == "__main__":
    article = 'LANGSA - Polres Langsa, Selasa (12/2) dini menangkap Husaini (55) warga Desa Pulo Baro Tangse, Pidie, ketika berupaya menyelundupkan 12 ekor trenggiling. <EOL> Sampai tadi malam pria tersebut bersama barang bukti 12 ekor binatang yang dilindungi itu masih diamankan di Mapolres setempat. <EOL> Kapolres Langsa, AKBP Hariadi SIK, melalui Kasat Reskrim AKP Muhammad Firdaus, kepada Serambi mengatakan, keberhasilan penangkapan penyelundup 12 ekor trenggiling itu setelah ada laporan dari masyarakat. <EOL> Dalam laporan masyarakat itu disebutkan bahwa seorang pria dari Tangse Pidie sedang membawa 12 ekor trenggiling ke Medan Husaini, sedang membawa sebanyak 12 ekor trenggiling tujuan ke Medan dengan menumpangi bus Kurnia. <EOL> Atas informasi tersebut, kata AKP Muhammad Firdaus, sejumlah anggota Satuan Reskrim Polres Langsa, Selasa (12/2) sekitar pukul 03.00 WIB menjelang pagi, memberhentikan bus yang sudah ketahui nomor polisinya, tepat di depan Mapolres Langsa. Katanya, setelah digeledah di bagian bagasi bus, ditemukan 12 ekor trenggiling yang berada dalam satu karung goni besar. <EOL> Kasat Reskrim menambahkan, dari 12 ekor trenggiling itu, satu diantaranya suda mati. Setelah meenurunkan barang bukti (BB) trenggiling itu, selanjutnya petugas mencari tahu pemilik barang terlarang itu.  <EOL> “Saat itu juga tersangka Husaini dan BB sebanyak 12 ekor trenggiling langsung diamankan ke Mapolres untuk mempertanggung jawabkan perbuatannnya,”ujarnya. Tersangka mengaku trenggiling itu akan dibawa ke Medan, dan di sana suda ada yang menunggu untuk membeli binatang dilindungi tersebut. <EOL> Menurut AKP Firdaus, atas perbuatannya itu tersangka Husaini dikenakan Pasal 21 ayat 1 dan 2 Jo Pasal 40 ayat 2 dan 4 Undang-Undang Nomor 5 Tahun 1990, tentang sumber daya hayati dan ekosistem dengan ancaman hukuman penjara lima tahun. Sementara itu sebanyak 12 ekor trenggiling dilindungi tersebut akan diserahkan kepada Badan Konservasi Sumber Daya Alam (BKSDA) Aceh.(c42) <EOS>'

    run(article)

