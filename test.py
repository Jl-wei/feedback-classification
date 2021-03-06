import logging
from datetime import datetime
from transformers import BertForSequenceClassification

from data_loader import train_test_dataloader
from utilities import read_json, get_device
from evaluate import evaluate_model


def main(config):
    name = f"{config['model_name']}-{config['data_file']}-test"
    logging.basicConfig(
        filename=f"./logs/{config['model_name']}-{config['data_file']}.log",
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    train_dataloader, val_dataloader = train_test_dataloader(config)

    model = BertForSequenceClassification.from_pretrained(f"./models/{config['load_model']}").to(get_device())

    eval_time_start = datetime.now()
    eval_report = evaluate_model(model, val_dataloader)
    eval_time = (datetime.now() - eval_time_start).total_seconds()

    training_stats = {
        "train_size": len(train_dataloader),
        "val_size": len(val_dataloader),
        "evaluation_report": eval_report,
        "eval_time": eval_time,
    }
    
    logging.info(f"Training Stats: \n {training_stats}")
    print(f"Evaluation Report: \n {eval_report}")
    
if __name__ == '__main__':
    config = read_json('./config.json')
    
    config['data_file'] = "P1-Golden"
    config['load_model'] = "BERT-P1-Golden"
    main(config)