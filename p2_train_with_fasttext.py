import random
import numpy as np
import os
import logging
import torch
import pandas as pd
from imp import reload
import pickle
import fasttext
from sklearn.metrics import classification_report
from datetime import datetime

reload(logging)

# Parameters
seed = 42
epochs = 25
batch_size = 16
learning_rate = 4e-5
golden_2 = pd.read_excel("./data/P2-Golden.xlsx")
VECTORS_FILEPATH = "./models/pre-trained/crawl-300d-2M.vec"
SAVE_MODEL = True
output_dir = "./models/"

# Set up log file
logging.basicConfig(
    filename=f"{os.getcwd()}/fasttext-p2.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

if torch.cuda.is_available():
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
# Shuffle the DataFrame rows
golden_2 = golden_2.sample(frac=1)

fasttext_input = []
fasttext_label = []
for i, row in golden_2.iterrows():
    text = row["reviews"]
    label = "__label__" + str(row["Judgement"])
    fasttext_label.append(label)
    fasttext_input.append(label + " " + text)

golden_2["fasttext_input"] = fasttext_input
golden_2["fasttext_label"] = fasttext_label

# Split the dataframe into 95:5 train:validation splits
train_file = open("./data/p2.fasttext.train", "w", encoding="utf-8")
validation_rows = []
ind = int((len(golden_2)*95)/100)
for i, row in golden_2.iterrows():
    if i < ind:
        train_file.write(row["fasttext_input"] + "\n")
    else:
        validation_rows.append((row["Judgement"], row["reviews"]))
train_file.close()


# ========================================
#               Training
# ========================================
training_stats = []
start_time = datetime.now()

model = fasttext.train_supervised(input="./data/p2.fasttext.train", lr=learning_rate, epoch=epochs,
                                  wordNgrams=2, bucket=200000, dim=300, loss='hs',
                                  pretrainedVectors=VECTORS_FILEPATH)
train_time = (datetime.now() - start_time).total_seconds()

# ========================================
#               Evaluation
# ========================================
predicted = []
y_test = []
eval_time_start = datetime.now()

for (l, r) in validation_rows:
    y_test.append(l)
    prediction = int(model.predict(r.strip("\n"))[0][0].split("_")[-1])
    predicted.append(prediction)

eval_report = classification_report(y_test, predicted, output_dict=True)
eval_time = (datetime.now() - eval_time_start)

training_stats.append(
    {"evaluation_report": eval_report, "train_time": train_time, "eval_time": eval_time}
)
logging.info(f"Training Stats: \n {training_stats}")
print(f"Evaluation Report: \n {eval_report}")

# Save report
with open("fasttext-p2.pkl", "wb") as f:
    pickle.dump(training_stats, f)

if SAVE_MODEL:
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving model to {output_dir}")

    # Save a trained model.
    # They can then be reloaded using fasttext.load_model(PATH)
    model.save_model(f"{output_dir}/p2-fasttext.bin")
