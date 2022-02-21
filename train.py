import random
import numpy as np
import pandas as pd
import torch
import logging
import os
import pickle
from imp import reload
from datetime import datetime
from transformers import AdamW, get_linear_schedule_with_warmup

from data_loader import train_test_dataloader
from models import get_model
from trainer import train_model
from utilities import get_device, read_json
from evaluate import evaluate_model


# Set the seed value all over the place to make this reproducible.
seed = 57
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    reload(logging)
    
    name = f"{config['model_name']}-{config['data_file']}"
    logging.basicConfig(
        filename=f"./logs/{config['model_name']}-{config['data_file']}.log",
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    
    train_dataloader, val_dataloader = train_test_dataloader(config)

    logging.info(f"Number of train samples: {len(train_dataloader)}")
    logging.info(f"Number of validation samples: {len(val_dataloader)}")
    
    start_time = datetime.now() # Measure the total training time for the whole run.
    model, stats = train(config, train_dataloader, val_dataloader)
    
    train_time = (datetime.now() - start_time).total_seconds()

    eval_time_start = datetime.now()
    eval_report = evaluate_model(model, val_dataloader)
    eval_time = (datetime.now() - eval_time_start).total_seconds()
    
    training_stats = {
        "train_size": len(train_dataloader),
        "val_size": len(val_dataloader),
        "training_stats": stats,
        "evaluation_report": eval_report,
        "train_time": train_time,
        "eval_time": eval_time,
    }
    
    logging.info(f"Training Stats: \n {training_stats}")
    print(f"Evaluation Report: \n {eval_report}")
    
    # Save report
    with open(f"./reports/{name}.pkl", "wb") as f:
        pickle.dump(training_stats, f)

    # Save model
    if config['save_model']:
        # Create output directory if needed
        if not os.path.exists('./models/'):
            os.makedirs('./models/')

        print(f"Saving model to {'./models/'}")

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(f"./models/{name}/")

def train(config, train_dataloader, val_dataloader):
    # model
    model = get_model(config['model_name'], num_labels = config['num_labels']).to(get_device())

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=config['epsilon'],  # args.adam_epsilon  - default is 1e-8.
    )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * config['epochs']

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value in run_glue.py
        num_training_steps=total_steps,
    )

    model, stats = train_model(
        model, train_dataloader, val_dataloader, optimizer, scheduler, seed=seed, epochs=config['epochs']
    )
    
    return model, stats

if __name__ == '__main__':
    config = read_json('./config.json')
    
    main(config)
    
    config['data_file'] = "P2-Golden"
    main(config)
    
    config['data_file'] = "P3-Golden"
    config['num_labels'] = 5
    main(config)
    
    config['data_file'] = "SHAH"
    config['num_labels'] = 4
    main(config)