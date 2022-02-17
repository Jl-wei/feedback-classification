import random
import numpy as np
import os
import logging
import torch
from utilities import get_device, current_utc_time
import pandas as pd
from imp import reload
from data_loader import get_loader, prepare_dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from models import get_model
from trainer import train_model
from evaluate import evaluate_model
import pickle
from datetime import datetime

reload(logging)

# Parameters
model_name = "BERT"
seed = 42
epochs = 25
batch_size = 16
learning_rate = 2e-5
epsilon = 1e-8
golden_2 = pd.read_excel("./data/P2-Golden.xlsx")
SAVE_MODEL = True
output_dir = "./models/"


# Set up log file
current_time = current_utc_time()
logging.basicConfig(
    filename=f"{os.getcwd()}/bert-p2.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

device = get_device()
# Set the seed value all over the place to make this reproducible.
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Prepare dataset
all_input_ids, all_attention_masks, all_labels = prepare_dataset(golden_2)

# Shuffle data and separate evaluation dataset
print("Separating test and train data")
while True:
    indices = np.arange(all_input_ids.shape[0])
    np.random.shuffle(indices)
    all_input_ids = all_input_ids[indices]
    all_attention_masks = all_attention_masks[indices]
    all_labels = all_labels[indices]
    val_labels = all_labels[:50]

    # Ensure that we do not have too much bias in validation dataset
    bias_ratio = np.count_nonzero(val_labels == 1) / np.count_nonzero(val_labels == 0)
    if 0.75 < bias_ratio < 1.25:
        val_input_ids = all_input_ids[:50]
        val_attention_masks = all_attention_masks[:50]
        break

val_dataloader = get_loader(
    val_input_ids, val_attention_masks, val_labels, loader_type="VALIDATE", batch_size=batch_size
)

input_ids = all_input_ids[50:]
attention_masks = all_attention_masks[50:]
labels = all_labels[50:]

logging.info(f"Number of train samples: {len(input_ids)}")
logging.info(f"Number of validation samples: {len(val_input_ids)}")

# Measure the total training time for the whole run.
start_time = datetime.now()

# ========================================
#               Training
# ========================================

# Prepare dataloader
train_dataloader = get_loader(input_ids, attention_masks, labels, batch_size=batch_size)

# model
model = get_model(model_name).to(device)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
    eps=epsilon,  # args.adam_epsilon  - default is 1e-8.
)

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,  # Default value in run_glue.py
    num_training_steps=total_steps,
)

model, stats = train_model(
    model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=epochs, seed=seed
)

# ========================================
#               Evaluation
# ========================================
train_time = (datetime.now() - start_time).total_seconds()
eval_time_start = datetime.now()
eval_report = evaluate_model(model, val_dataloader)
eval_time = (datetime.now() - eval_time_start).total_seconds()

training_stats = {
    "train_size": len(labels),
    "val_size": len(val_labels),
    "training_stats": stats,
    "evaluation_report": eval_report,
    "train_time": train_time,
    "eval_time": eval_time,
}

logging.info(f"Training Stats: \n {training_stats}")
print(f"Evaluation Report: \n {eval_report}")

# Save report
with open("bert-p2.pkl", "wb") as f:
    pickle.dump(training_stats, f)

if SAVE_MODEL:
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving model to {output_dir}")

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
