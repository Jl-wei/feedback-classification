import random
import numpy as np
import os
import logging
import torch
import pandas as pd
from imp import reload
import pickle
from sklearn.metrics import classification_report
from models import get_model
from utilities import Corpus, batchify
from trainer import train_elmo
from evaluate import evaluate_elmo
import collections
import torch.optim as optim
import torch.nn as nn
from datetime import datetime

reload(logging)

# Parameters
seed = 42
epochs = 25
batch_size = 16
learning_rate = 2e-4
golden_1 = pd.read_excel("./data/P1-Golden.xlsx")
CUDA = torch.cuda.is_available()
SAVE_MODEL = True
output_dir = "./models/"

# Set up log file
logging.basicConfig(
    filename=f"{os.getcwd()}/elmo-p1.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

if CUDA:
    device = torch.device("cuda")
    logging.info(f"There are {torch.cuda.device_count()} GPU(s) available.")
    logging.info(f"We will use the GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.info("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

# Set the seed value all over the place to make this reproducible.
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Prepare dataset

corpus = Corpus(sentences=list(golden_1.reviews), labels=list(golden_1.Judgement))
# Print corpus stats
class_counts = collections.Counter([c[0] for c in corpus.train])
print("Train: {}".format(class_counts))
class_counts = collections.Counter([c[0] for c in corpus.test])
print("Test: {}".format(class_counts))

train_data = batchify(corpus.train, batch_size, shuffle=True)
test_data = batchify(corpus.test, batch_size, shuffle=False)

print("Loaded data!")

# Model
model = get_model("ELMO").to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Scheduler
learning_rate_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Loss
criterion_ce = nn.CrossEntropyLoss()

# Initialize weights
model.init_weights()

start_time = datetime.now()

training_stats = []
niter_global = 0
for epoch in range(1, epochs + 1):
    logging.info(f"Epoch {epoch}/{epochs}...")
    print(f"Epoch {epoch}/{epochs}...")

    # ========================================
    #               Training
    # ========================================
    # loop through all batches in training data
    train_loss = 0
    nbatches = 0
    for train_batch in train_data:
        loss = train_elmo(model, train_batch, optimizer, criterion_ce, gpu=CUDA)
        train_loss += loss
        niter_global += 1
        nbatches += 1
        if niter_global % 10 == 0:
            msg = "Train loss {:.5f}".format(loss)
            print(msg)
            logging.info(msg)

    train_loss = train_loss / nbatches

    # ========================================
    #               Validation
    # ========================================
    with torch.no_grad():
        accuracy, v_loss = evaluate_elmo(model, test_data, criterion_ce, gpu=CUDA)
    msg = "val acc {:.4f}, val loss {:.4f}".format(accuracy, v_loss)
    print(msg)
    logging.info(msg)
    # we use generator, so must re-gen test data
    test_data = batchify(corpus.test, batch_size, shuffle=False)

    # clear cache between epoch
    torch.cuda.empty_cache()
    # decay learning rate
    learning_rate_scheduler.step()
    # shuffle between epochs
    train_data = batchify(corpus.train, batch_size, shuffle=True)

    training_stats.append(
        {
            "epoch": epoch,
            "Training Loss": train_loss,
            "Valid. Loss": v_loss,
            "Valid. Accur.": accuracy,
        }
    )

train_time = (datetime.now() - start_time).total_seconds()

# ========================================
#               Evaluation
# ========================================
model.eval()
y_test = []
predictions = []
eval_time_start = datetime.now()

for batch in test_data:
    source, labels = batch
    y_test += labels.tolist()
    if CUDA:
        source = source.to("cuda")
    output = model(source)
    _, max_indices = torch.max(output, -1)
    predictions += max_indices.tolist()

eval_report = classification_report(y_test, predictions, output_dict=True)
eval_time = (datetime.now() - eval_time_start).total_seconds()

training_stats.append({"evaluation_report": eval_report, "train_time": train_time, "eval_time": eval_time,})
logging.info(f"Training Stats: \n {training_stats}")
print(f"Evaluation Report: \n {eval_report}")

# Save report
with open("elmo-p1.pkl", "wb") as f:
    pickle.dump(training_stats, f)

if SAVE_MODEL:
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving model to {output_dir}")

    # Save a trained model.
    # They can then be reloaded using:
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    torch.save(model.state_dict(), f"{output_dir}/p1-elmo.bin")
