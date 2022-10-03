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
        self.loss_fn = nn.CrossEntropyLoss()

        # if testing is set True, 
        # then batch['intent'] is not provided
        self.testing = False

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.hidden_size * (2 if self.bidirectional else 1)
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        # batch is the result of collate_fn in dataset.py 
        x = torch.tensor(batch['text'])
        
        # Improve training efficiency with pack_padded_sequence
        embed = self.embed(x)
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, torch.tensor(batch['seq_len']), batch_first=True)

        _, (h, _) = self.rnn(packed_embed)
        if self.bidirectional:
            h = torch.cat((h[-1], h[-2]), 1) # [batch_size, 2*hidden_size]
        else:
            h = h[-1]  # [batch_size, hidden_size]

        out = self.classifier(h) # [batch_size, num_class]

        # Classify each text to the intent with max probability
        pred = out.max(dim=1)[1]

        res = { 
          'out': out,    # shape: (batch_size, num_class)
          'pred': pred,  # Classfication result, shape: (batch_size)
        }

        if not self.testing:
            y = torch.tensor(batch['intent']).long()
            res['ok'] = (pred == y).sum().item()
            res['overall'] = len(x)
            res['loss'] = self.loss_fn(out, y)

        return res
        
        







class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
