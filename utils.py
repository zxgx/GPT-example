import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

import colossalai


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src  # batch size, src seq len
        self.src_mask = (src != pad).unsqueeze(-2)  # batch size, 1, src seq len

        if trg is not None:
            self.trg = trg[:, :-1]   # batch size, trg seq len - 1
            self.trg_y = trg[:, 1:]  # batch size, trg seq len - 1
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).sum().item()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)  # batch size, 1, trg seq len - 1
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask


def slurm_init():
    parser = colossalai.get_default_parser()
    parser.add_argument('--config_path', default='./config.py', type=str)

    args = parser.parse_args()

    # HOST=$(scontrol show hostname $SLURM_NODELIST | head -n1)
    colossalai.launch_from_slurm(config=args.config_path, host=args.host, port=args.port)


class WebtextDataset(Dataset):
    def __init__(self, path, seq_len=1024, cache_dir=None) -> None:
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2' if cache_dir is None else cache_dir)
        self.tokenizer.pad_token = self.tokenizer.unk_token

        root = os.path.dirname(path)
        encoded_data_cache_path = os.path.join(root, f'gpt_webtext_{seq_len}.pt')
        if os.path.isfile(encoded_data_cache_path):
            seq_len_, data, attention_mask = torch.load(encoded_data_cache_path)
            if seq_len_ == seq_len:
                self.data = data
                self.attention_mask = attention_mask
                return
        raw_data = []
        with open(path) as f:
            for line in f.readlines():
                raw_data.append(json.loads(line)['text'])
        encoded_data = self.tokenizer(raw_data, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
        self.data = encoded_data['input_ids']
        self.attention_mask = encoded_data['attention_mask']
        torch.save((seq_len, self.data, self.attention_mask), encoded_data_cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'input_ids': self.data[index],
            'attention_mask': self.attention_mask[index]
        }, self.data[index]
