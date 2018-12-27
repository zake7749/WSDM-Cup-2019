"""ARCT specific model."""
import torch
from torch import nn
from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from pytorch_pretrained_bert import BertModel


class ARCTModel(PreTrainedBertModel):

    def __init__(self, config, num_labels=2):
        super(ARCTModel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids_0, token_type_ids_0=None, attention_mask_0=None,
                input_ids_1=None, token_type_ids_1=None, attention_mask_1=None,
                labels=None):
        _, pooled_output_0 = self.bert(
            input_ids_0, token_type_ids_0, attention_mask_0,
            output_all_encoded_layers=False)
        pooled_output_0 = self.dropout(pooled_output_0)

        _, pooled_output_1 = self.bert(
            input_ids_1, token_type_ids_1, attention_mask_1,
            output_all_encoded_layers=False)
        pooled_output_1 = self.dropout(pooled_output_1)

        logits_0 = self.classifier(pooled_output_0)
        logits_1 = self.classifier(pooled_output_1)

        logits = torch.cat([logits_0, logits_1], dim=1)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


class ARCT2(PreTrainedBertModel):

    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768 * 3, 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, cw0_i, cw0_m, cw0_s, cw1_i, cw1_m, cw1_s,
                rw0_i, rw0_m, rw0_s, rw1_i, rw1_m, rw1_s,
                w0w1_i, w0w1_m, w0w1_s, labels=None):
        c_w0 = self.bert_forward(cw0_i, cw0_m, cw0_s)
        c_w1 = self.bert_forward(cw1_i, cw1_m, cw1_s)
        r_w0 = self.bert_forward(rw0_i, rw0_m, rw0_s)
        r_w1 = self.bert_forward(rw1_i, rw1_m, rw1_s)
        w0_w1 = self.bert_forward(w0w1_i, w0w1_m, w0w1_s)
        features0 = torch.cat([c_w0, r_w0, w0_w1], dim=1)
        features1 = torch.cat([c_w1, r_w1, w0_w1], dim=1)
        scores0 = self.classifier(features0)
        scores1 = self.classifier(features1)
        logits = torch.cat([scores0, scores1], dim=1)
        loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def bert_forward(self, i, m, s, pooled=True):
        # input_ids, token_ids, attention_mask
        vecs, pooled_output = self.bert(
            i, m, s, output_all_encoded_layers=False)
        if pooled:
            return self.dropout(pooled_output)
        else:
            return self.dropout(vecs)
