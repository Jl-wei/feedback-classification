from transformers import BertForSequenceClassification
import logging
import torch.nn as nn
from allennlp.modules.elmo import Elmo


def get_model(name: str = "BERT", **kwargs):

    if name == "BERT":
        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classification layer on top.
        num_labels = kwargs.get("num_labels", 2)
        output_attentions = kwargs.get("output_attentions", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=output_attentions,  # Whether the model returns attentions weights.
            output_hidden_states=output_hidden_states,  # Whether the model returns all hidden-states.
        )
        params = list(model.named_parameters())
        logging.info(
            "The BERT model has {:} different named parameters.\n".format(len(params))
        )

        logging.info("==== Embedding Layer ====\n")

        for p in params[0:5]:
            logging.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        logging.info("\n==== First Transformer ====\n")

        for p in params[5:21]:
            logging.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        logging.info("\n==== Output Layer ====\n")

        for p in params[-4:]:
            logging.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        return model

    elif name == "ELMO":
        num_labels = kwargs.get("num_labels", 2)
        on_gpu = kwargs.get("on_gpu", True)
        dropout = kwargs.get("dropout", 0.5)
        model = SimpleELMOClassifier(
            label_size=num_labels, use_gpu=on_gpu, dropout=dropout
        )
        return model

    return None


class SimpleELMOClassifier(nn.Module):
    def __init__(self, label_size, use_gpu, dropout):
        super(SimpleELMOClassifier, self).__init__()
        self.use_gpu = use_gpu
        self.dropout = dropout
        options_file = "./models/pre-trained/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
        weight_file = "./models/pre-trained/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
        self.elmo = Elmo(
            options_file, weight_file, 1, dropout=dropout, do_layer_norm=False
        )
        # elmo output
        #         Dict with keys:
        #         ``'elmo_representations'``: ``List[torch.Tensor]``
        #             A ``num_output_representations`` list of ELMo representations for the input sequence.
        #             Each representation is shape ``(batch_size, timesteps, embedding_dim)``
        #         ``'mask'``:  ``torch.Tensor``
        #             Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
        self.conv1 = nn.Conv1d(1024, 16, 3)
        self.p1 = nn.AdaptiveMaxPool1d(128)
        self.activation_func = nn.ReLU6()
        self.dropout_l = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(2048, label_size)

    def init_weights(self):
        for name, param in self.hidden2label.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)
        for name, param in self.conv1.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, sentences):
        elmo_out = self.elmo(sentences)
        x = elmo_out["elmo_representations"][0]
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation_func(x)
        x = self.p1(x)
        x = x.view(-1, 2048)
        x = self.dropout_l(x)
        y = self.hidden2label(x)
        return y
