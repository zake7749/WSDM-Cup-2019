"""WSDM specific models."""
from torch import nn
from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from pytorch_pretrained_bert import BertModel
from bert import util


class PseudoLabelModel(PreTrainedBertModel):

    def __init__(self, config, num_labels=3):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 3)
        self.apply(self.init_bert_weights)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = util.SoftCrossEntropy()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = self.criterion(logits.view(-1, self.num_labels), labels.float())
            return loss, logits
        else:
            return logits
