from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)
    
    # __get_item__(idx) collects some data points and pack them into a list
    # which was then sent into collate_fn
    # Iterating the dataloader returns the result of collate_fn
    #
    # returns a dictionary batch, where
    # batch['text'] = text data in integer ids, shape: (batch_size, max(seq_len))
    # batch['intent'] = labels of each text data each in integer id, shape: (batch_size, )
    # batch['seq_len'] = list of seq_len of each text data
    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        
        batch = {}
        samples.sort(key=lambda x: len(x['text'].split()), reverse=True)

        # X = batch['text]
        batch['text'] = [s['text'].split() for s in samples]

        # in order to use nn.utils.rnn.pack_padded_sequence
        batch['seq_len'] = [len(x) for x in batch['text']]

        # convert 2d string to 2d integer ids
        batch['text'] = self.vocab.encode_batch(batch['text'])

        # Y = batch['intent']
        batch['intent'] = [self.label2idx(s['intent']) for s in samples]

        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        raise NotImplementedError
