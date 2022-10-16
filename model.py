from typing import Dict

import torch
import torch.nn as nn

import os

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
          dropout=dropout,
          batch_first = True,  # size = [batch_size, seq_len, embed_dim]
        )

        # 3. FC layer
        self.classifier = nn.Sequential(
          nn.Dropout(dropout),
          nn.Linear(self.encoder_output_size, num_class),
        )

        # We use cross entropy loss as our default loss function
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
        self.rnn.flatten_parameters()
        
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
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        use_crf: bool
    ) -> None:
        super(SeqTagger, self).__init__(embeddings, hidden_size, num_layers, dropout, bidirectional, num_class)
        
        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(self.encoder_output_size, num_class, dropout)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        self.rnn.flatten_parameters()
        
        # TODO: implement model forward
        x = torch.tensor(batch['tokens'])
        batch['seq_len'] = torch.tensor(batch['seq_len'])

        # Embedding
        embed = self.embed(x) # (batch_size, max_len, embed_dim)
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, batch['seq_len'], batch_first=True)

        # LSTM
        out, (_, _) = self.rnn(packed_embed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True) # TODO: (batch_size, max_len, encoder_output_size)

        # Align with max_len
        mask = batch['mask'][:, :out.size(1)]

        if not self.use_crf:
            out = self.classifier(out)          # (batch_size, max_len, num_class)
            pred = out.max(dim=-1)[1]           # (batch_size, max_len)
        else:
            pred = self.crf(out, mask)          # (batch_size, max_len)

        res = {
          'pred': pred,
        }

        # calculate loss
        if not self.testing:
            # Align with max_len
            y = batch['tags'].long()[:, :out.size(1)]

            if not self.use_crf:
                # cross entropy loss
                # input: (batch_size, num_class, seq_len)
                # target: (batch_size, num_class)
                res['loss'] = self.loss_fn(out.permute(0, 2, 1), y)
            else:
                res['loss'] = self.crf.nll_loss(out, y, mask)

        return res


# Framework from https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
class CRF(nn.Module):
    # OK
    def __init__(self, input_dim, num_class, dropout):
        super(CRF, self).__init__()
        self.num_class = num_class + 2      # add start_tag and stop_tag
        self.start_idx = self.num_class - 2  # start_tag
        self.stop_idx = self.num_class - 1  # stop_tag

        self.fc = nn.Sequential(
          nn.Dropout(dropout),
          nn.Linear(input_dim, self.num_class),
        )

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning from i to j
        self.transitions = nn.Parameter(
            torch.randn(self.num_class, self.num_class), requires_grad=True)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.start_idx, :] = -1
        self.transitions.data[:, self.stop_idx] = -1

    # Helper function
    # x: (batch_size, self.num_class, self.num_class) or (batch_size, self.num_class) 
    def log_sum_exp(self, x, dim):
        return torch.logsumexp(x, dim=dim)

    # OK
    # Compute total score of all paths
    # x: (batch_size, max_len, self.num_class)
    def _forward_alg(self, x, mask):
        batch_size, max_len, _ = x.shape
        
        # scores shape: (batch_size, self.num_class)
        # scores[b][c] = 
        # At some time step l, compressed total score of paths ending at class c 
        scores = torch.full_like(x[:, 0, :], -1e5)

        for l in range(max_len):
            mask_l = mask[:, l:l+1] # (batch_size, 1)
            
            prev_scores = scores.unsqueeze(-1)                # (batch_size, self.num_class, 1)
            emission_scores = x[:, l, :].unsqueeze(1)         # (batch_size, 1, self.num_class)
            transition_scores = self.transitions.unsqueeze(0) # (1, self.num_class, self.num_class)
            
            new_scores = self.log_sum_exp(prev_scores + emission_scores + transition_scores, -2) # (batch_size, self.num_class)
            scores = new_scores * mask_l + scores * (~mask_l) # update score for tokens that has word i
        
        # Transition score from last tag to stop_tag
        # LHS = (batch_size, num_class)
        # RHS = (1, num_class)
        scores += self.transitions[:, self.stop_idx].unsqueeze(0)

        return self.log_sum_exp(scores, dim=-1).unsqueeze(-1)   # (batch_size, 1)

    # OK
    def _score_sentence(self, x, y, mask):
        # Gives the score of `batch_size` provided tag sequence
        # x: (batch_size, max_len, num_class) from bilstm layer
        # y: (batch_size, max_len)
        # mask: (batch_size, max_len)

        batch_size, max_len, _ = x.shape

        # Emission Score
        # torch.gather reference: https://reurl.cc/QbMy1M  
        emission_score = (x.gather(-1, y.unsqueeze(-1))).squeeze() # (batch_size, max_len)
        
        # Transition Score
        # start_tag -> tag1 -> tag2 -> ... -> last tag
        start_tag_column = torch.full((batch_size, 1), fill_value=self.start_idx, device=x.device) 
        y_augmented = torch.cat([start_tag_column, y], dim=1) # (batch_size, max_len+1)
        transition_score = self.transitions[y_augmented[:, :-1], y_augmented[:, 1:]] # (batch_size, max_len)
        
        # Transition score
        # last tag -> stop_tag
        seq_len = mask.sum(dim=-1, keepdim=True).long() # (batch_size, 1)
        last_tag_idxs = y_augmented.gather(-1, seq_len) # (batch_size, 1)
        last_transition_score = self.transitions[last_tag_idxs, self.stop_idx] # (batch_size, 1)

        score = (mask * (emission_score + transition_score)).sum(-1, keepdims=True) + last_transition_score

        return score  # (batch_size, 1)

    # OK
    # start_tag -> w0 -> w1 -> ... -> w_max_len -> stop_tag
    # scores = tran_score(start_tag->w0) + tran_score(w0->w1) + ... + tran_score(w_max_len -> stop_tag) +
    #          emi_score(w0) + emi_score(w1) + ... + emi_score(w_max_len)
    @torch.no_grad()
    def _viterbi_decode(self, x, mask):
        batch_size, max_len, _ = x.shape
        
        # Use zeros_like instead of zeros to make sure all tensors are on the same device
        best_scores = torch.zeros_like(x[:, 0, :]) # (batch_size, self.num_class)
        backtrack = torch.zeros_like(x).long() # (batch_size, max_len, self.num_class)
        
        for l in range(max_len):
            prev = best_scores.unsqueeze(-1)         # (batch_size, self.num_class, 1)
            trans = self.transitions.unsqueeze(0)    # (1, self.num_class, self.num_class)
            new_score = prev + trans                 # (batch_size, self.num_class, self.num_class)
            
            # Find best score of path w/ length l whose end is label i (so we take max over dim=-2) 
            # new_score: (batch_size, self.num_class)
            new_score, backtrack[:, l, :] = new_score.max(dim=-2)
            
            # new_score[b, i] = best score of path with length l ending at label i
            # so we need to add the emission score of word l being labeled as label i
            emi = x[:, l, :]
            new_score += emi

            mask_l = mask[:, l].unsqueeze(-1)  # (batch_size, 1)
            best_scores = mask_l * new_score + (~mask_l) * best_scores
        
        # tran_score(w_max_len -> stop_tag)
        best_scores += self.transitions[:, self.stop_idx].unsqueeze(0)

        # Find the last tag for each best path in the batch
        _, best_last_tags = best_scores.max(dim=-1)     # (batch_size)

        seq_len = mask.sum(dim=-1).long()               # (batch_size)

        # Backtrack to find the best path
        best_paths = []
        for b in range(batch_size):
            tokens_len = seq_len[b].item()

            cur_tag = best_last_tags[b].item()
            bp = [cur_tag] if tokens_len > 0 else []
            
            for l in range(tokens_len-1, 0, -1):
                prev_tag = backtrack[b, l, cur_tag].item()
                bp.append(prev_tag)
                cur_tag = prev_tag
            
            bp.reverse()
            bp += [0] * (max_len - tokens_len)
            
            best_paths.append(bp)
        
        best_paths = torch.tensor(best_paths)
        return best_paths

    # OK
    # ref: https://zhuanlan.zhihu.com/p/44042528
    def nll_loss(self, x, y, mask):
        # x: (batch_size, max_len, hidden_size)
        # y: (batch_size, max_len)

        # emission matrix
        x = self.fc(x)  # (batch_size, max_len, self.num_class)

        forward_score = self._forward_alg(x, mask)    # (batch_size, 1)
        gold_score = self._score_sentence(x, y, mask) # (batch_size, 1)

        return (forward_score - gold_score).mean()
       

    # OK
    def forward(self, x, mask):  # for prediction
        x = self.fc(x)
        return self._viterbi_decode(x, mask)
        