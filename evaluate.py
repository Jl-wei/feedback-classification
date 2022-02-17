from utilities import get_device
import torch
import numpy as np
from sklearn.metrics import classification_report
import logging

device = get_device()


def evaluate_model(model, dataloader):
    model.eval()

    logging.info("Starting Evaluation...")
    print("Starting Evaluation...")

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        # Move logits and labels to CPU
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    logging.info("    DONE.")
    print("    DONE.")

    truth = np.hstack((true_labels))
    a = np.vstack((predictions))
    pred = np.argmax(a, axis=1)
    logging.info(f"Prediction: {pred}")
    logging.info(f"Truth: {truth}")

    report = classification_report(truth, pred)

    logging.info(f"Classification Report: \n {report}")
    print(f"Classification Report: \n {report}")

    return report


def evaluate_elmo(model, data, criterion_ce, gpu=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    all_accuracies = 0.0
    all_loss = 0.0
    nbatches = 0.0

    for batch in data:
        nbatches += 1.0
        source, tags = batch
        if gpu:
            source = source.to("cuda")
            tags = tags.to("cuda")
        # output = model(source, lengths)
        output = model(source)
        v_loss = criterion_ce(output, tags)
        max_vals, max_indices = torch.max(output, -1)

        accuracy = torch.mean(max_indices.eq(tags).float()).item()
        all_accuracies += accuracy
        all_loss += v_loss
    return all_accuracies / nbatches, all_loss / nbatches
