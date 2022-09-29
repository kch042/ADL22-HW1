from typing import Dict

import torch
import torch.nn as nn

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()

        # 0. save hyper parameters
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_class = num_class
        
        # 1. embedding layer
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)
        
        # 2. LSTM layer
        self.embed_size = embeddings.shape[1]
        self.rnn = nn.LSTM(
          input_size = self.embed_size, 
          hidden_size = hidden_size,
          num_layers = num_layers,
          bidirectional = bidirectional,
          batch_first = True,  # size = [batch_size, seq_len, embed_dim]
          )

        # 3. Linear layer
        self.classifier = nn.Sequential(
          nn.Dropout(dropout),
          nn.Linear(self.encoder_output_size, num_class),
        )

        # We use cross entropy loss as our loss function
        # so we don't need a final SM layer
        
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.hidden_size * (2 if self.bidirectional else 1)
    
    # batch is the result of collate_fn in dataset.py 
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward

        ## pack_padded_sequence
        x = batch['text']

        batch_size = len(x)
        max_seq_len = len(x[0])
        
        out = self.embed(x)
        assert(out.shape == (batch_size, max_seq_len, self.embed_size), 
          f'embed size error:\nwant: ({batch_size}, {max_seq_len}, {self.embed_size})\ngot: {out.shape}'
        )

        _, (h, _) = self.rnn(out)
        if self.bidirectional:
            h = torch.cat((h[-1], h[-2]), 1) # [batch_size, 2*hidden_size]
        else:
            h = h[-1]  # [batch_size, hidden_size]

        pred = self.classifier(h)
        assert(pred.shape == (batch_size, self.num_class), f'pred size error:\nwant: ({batch_size}, {self.num_class})\ngot: {pred.shape}')
        
        
        







class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
