from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import logging
import pandas as pd


def prepare_dataset(dataset):
    sentences = dataset.reviews.values
    labels = np.array(dataset.Judgement)

    # Load the BERT tokenizer.
    logging.info("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    logging.info(f"Original: {sentences[0]}")
    logging.info(f"Tokenized: {tokenizer.tokenize(sentences[0])}")
    logging.info(
        f"Token IDs: {tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0]))}"
    )

    max_len = 0
    # For every sentence...
    for sent in sentences:
        input_ids = tokenizer.encode(str(sent), add_special_tokens=True)
        max_len = max(max_len, len(input_ids))

    logging.info(f"Max sentence length: {max_len}")

    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len + 10,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
            truncation=True,
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict["input_ids"])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict["attention_mask"])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Print sentence 0, now as a list of IDs.
    logging.info(f"Original: {sentences[0]}")
    logging.info(f"Token IDs:{input_ids[0]}")
    logging.info(f"Length of labels: {len(labels)}")

    return input_ids, attention_masks, labels


def get_loader(input_ids, attention_masks, labels, batch_size=32, loader_type="TRAIN"):
    """
    Returns a dataloader for the dataset.

    The DataLoader needs to know our batch size for training.
    For fine-tuning BERT on a specific task, the authors recommend a batch
    size of 16 or 32.
    """
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    if loader_type == "TRAIN":
        # We'll take training samples in random order.
        dataloader = DataLoader(
            dataset,  # The training samples.
            sampler=RandomSampler(dataset),  # Select batches randomly
            batch_size=batch_size,  # Trains with this batch size.
        )
    else:
        # For validation the order doesn't matter, so we'll just read them sequentially.
        dataloader = DataLoader(
            dataset,  # The validation samples.
            sampler=SequentialSampler(dataset),  # Pull out batches sequentially.
            batch_size=batch_size,  # Evaluate with this batch size.
        )
    return dataloader


def train_test_dataloader(config):
    data_file = pd.read_excel(f"./data/{config['data_file']}.xlsx")
    
    # Separate evaluation dataset
    train, test = train_test_split(data_file, test_size=config['test_size'], stratify = np.array(data_file[config['label_column']]))
    
    # Prepare dataset
    input_ids, attention_masks, labels = prepare_dataset(train)
    val_input_ids, val_attention_masks, val_labels = prepare_dataset(test)

    train_dataloader = get_loader(input_ids, attention_masks, labels, batch_size=config['batch_size'])
    val_dataloader = get_loader(
        val_input_ids, val_attention_masks, val_labels, batch_size=config['batch_size'], loader_type="VALIDATE"
    )
    
    return train_dataloader, val_dataloader
