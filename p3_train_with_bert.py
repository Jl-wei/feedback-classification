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
from train import train_model
from evaluate import evaluate_model
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split

reload(logging)

# Parameters
model_name = "BERT"
seed = 57
epochs = 15
batch_size = 8
learning_rate = 2e-4
epsilon = 1e-8
golden_3 = pd.read_excel("./data/P3-Golden.xlsx")
SAVE_MODEL = True
output_dir = "./models/"

# Set up log file
current_time = current_utc_time()
logging.basicConfig(
    filename=f"{os.getcwd()}/bert-p3.log",
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

# Separate evaluation dataset
train, test = train_test_split(golden_3, test_size=0.2)

# Prepare dataset
input_ids, attention_masks, labels = prepare_dataset(train, string_label = True)
val_input_ids, val_attention_masks, val_labels = prepare_dataset(test, string_label = True)

val_dataloader = get_loader(
    val_input_ids, val_attention_masks, val_labels, batch_size=batch_size, loader_type="VALIDATE"
)

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
model = get_model(model_name, num_labels = 5).to(device)

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
    model, train_dataloader, val_dataloader, optimizer, scheduler, seed=seed, epochs=epochs
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
with open("bert-p1.pkl", "wb") as f:
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
