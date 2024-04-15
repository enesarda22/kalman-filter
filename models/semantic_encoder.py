import torch
from torch import nn
from utils.general import get_device

from transformers import AutoTokenizer, AutoModel


class SemanticEncoder(nn.Module):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.device = get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name)

    def forward(self, m):
        tokens = self.tokenizer(
            m,
            padding="max_length",
            max_length=self.block_size,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        encoder_lhs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"]

        encoder_output = self.mean_pooling(
            bert_lhs=encoder_lhs,
            attention_mask=attention_mask,
        )

        return encoder_output[:, None, :]

    @staticmethod
    def mean_pooling(bert_lhs, attention_mask=None):
        if attention_mask is None:
            out = torch.mean(bert_lhs, 1)

        else:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(bert_lhs.size()).float()
            )

            out = torch.sum(bert_lhs * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        return out
