import torch
from torch import nn
from utils.general import get_device

from transformers import AutoModel


class SemanticEncoder(nn.Module):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, block_size, dataloader):
        super().__init__()
        self.block_size = block_size
        self.device = get_device()

        self.bert = AutoModel.from_pretrained(self.model_name)
        self.bert.embeddings.word_embeddings.weight = nn.Parameter(
            self.bert.embeddings.word_embeddings.weight[
                [dataloader.idx_to_id[i] for i in range(dataloader.vocab_size)], :
            ]
        )

    def forward(self, idx):
        encoder_lhs = self.bert(input_ids=idx)["last_hidden_state"]
        encoder_output = self.mean_pooling(bert_lhs=encoder_lhs)

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
