import numpy as np
import datetime
import torch
from dateutil import tz
import logging
import os
import random
from nltk.tokenize import WhitespaceTokenizer
from allennlp.modules.elmo import batch_to_ids
from transformers import BertTokenizer
import json
from pathlib import Path
from collections import OrderedDict


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"There are {torch.cuda.device_count()} GPU(s) available.")
        logging.info(f"We will use the GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    return device


def current_utc_time():
    return datetime.datetime.now().astimezone(tz.tzutc())

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


# https://github.com/ylhsieh/pytorch-elmo-classification
class Corpus(object):
    def __init__(
        self,
        sentences,
        labels,
        lowercase=False,
        test_size=0.05,
        max_len=-1,
        label_dict=None,
        shuffle=True,
    ):
        self.shuffle = shuffle
        self.lowercase = lowercase
        self.all_sentences = sentences
        self.all_labels = labels
        self.max_len = max_len
        self.label_dict = label_dict
        self.test_size = test_size
        self.test_sentences = []
        self.test_labels = []
        self.train_sentences = []
        self.train_labels = []

        self.split_dataset()
        self.tokenizer = WhitespaceTokenizer()

        self.train = self.tokenize(self.train_sentences, self.train_labels)
        self.test = self.tokenize(self.test_sentences, self.test_labels)

    def split_dataset(self):
        if self.shuffle:
            data = list(zip(self.all_sentences, self.all_labels))
            random.shuffle(data)
            self.all_sentences, self.all_labels = zip(*data)

        ind = int(self.test_size * len(self.all_sentences))
        self.test_sentences = self.all_sentences[:ind]
        self.test_labels = self.all_labels[:ind]
        self.train_sentences = self.all_sentences[ind:]
        self.train_labels = self.all_labels[ind:]

    def tokenize(self, sentences, labels):
        processed_sentences = []
        processed_labels = []
        cropped = 0
        for (sent, label) in zip(sentences, labels):
            if self.lowercase:
                sent = sent.lower().strip()
            else:
                sent = sent.strip()
            sent = self.tokenizer.tokenize(sent)
            if self.max_len > 0:
                if len(sent) > self.max_len:
                    cropped += 1
                sent = sent[: self.max_len]
            if self.label_dict:
                label = self.label_dict[label]
            processed_sentences.append(sent)
            processed_labels.append(label)
        print(f"Number of sentences cropped: {cropped}")

        return list(zip(processed_labels, processed_sentences))


def batchify(data, bsz, shuffle=False, gpu=False):
    if shuffle:
        random.shuffle(data)
    tags, sents = zip(*data)
    nbatch = (len(sents) + bsz - 1) // bsz
    # downsample biggest class
    # sents, tags = balance_tags(sents, tags)

    for i in range(nbatch):

        batch = sents[i * bsz : (i + 1) * bsz]
        batch_tags = tags[i * bsz : (i + 1) * bsz]
        # lengths = [len(x) for x in batch]
        # sort items by length (decreasing)
        # batch, batch_tags, lengths = length_sort(batch, batch_tags, lengths)

        # Pad batches to maximum sequence length in batch
        # find length to pad to

        # maxlen = lengths[0]
        # for b_i in range(len(batch)):
        #     pads = [pad_id] * (maxlen-len(batch[b_i]))
        #     batch[b_i] = batch[b_i] + pads
        # batch = torch.tensor(batch).long()
        batch = batch_to_ids(batch)
        batch_tags = torch.tensor(batch_tags).long()
        # lengths = [torch.tensor(l).long() for l in lengths]

        # yield (batch, batch_tags, lengths)
        yield batch, batch_tags
