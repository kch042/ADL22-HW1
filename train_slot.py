import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Tuple, List

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

from seqeval.scheme import IOB2
from seqeval.metrics import classification_report
from seqeval.metrics import precision_score, accuracy_score

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

# evaluate the accuracy
def evaluate(y_pred: List[List[str]], y_true: List[List[str]]):
    print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred)

def train_per_epoch(dataloader, model, optimizer, clip):
    model.train()

    loss_list = []

    y_pred, y_true = [], []
    for batch in dataloader:
        # forward pass
        batch['tokens'] = torch.tensor(batch['tokens']).to(args.device)
        batch['mask'] = batch['tokens'].gt(0).bool()
        batch['tags'] = torch.tensor(batch['tags']).to(args.device)
        output_dict = model(batch)
        
        # backward pass
        loss = output_dict['loss']
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        loss_list.append(loss.detach())

        # store the result
        for yp, yl, yt in zip(output_dict['pred'], batch['seq_len'], batch['tags']):
            y_pred.append(yp[:yl].detach().tolist())
            y_true.append(yt[:yl].detach().tolist())

    return y_pred, y_true

@torch.no_grad()
def validate_per_epoch(dataloader, model):
    model.eval()

    y_pred = []
    y_true = []
    for batch in dataloader:
        batch['tokens'] = torch.tensor(batch['tokens']).to(args.device)
        batch['mask'] = batch['tokens'].gt(0).bool()
        batch['tags'] = torch.tensor(batch['tags']).to(args.device)
        output_dict = model(batch)

        for yp, yl, yt in zip(output_dict['pred'], batch['seq_len'], batch['tags']):
            y_pred.append(yp[:yl].detach().tolist())
            y_true.append(yt[:yl].detach().tolist())

    return y_pred, y_true

def main(args):
    # Data 
    with open(args.cache_dir / 'vocab.pkl', 'rb') as f:
        vocab: Vocab = pickle.load(f)
    
    tag2idx: Dict[str, int] = json.loads((args.cache_dir / 'tag2idx.json').read_text())

    data_paths = {split: args.data_dir / f'{split}.json' for split in SPLITS}
    data = {split: json.loads(data_path.read_text()) for split, data_path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {split: SeqTaggingClsDataset(dat, vocab, tag2idx, args.max_len) for split, dat in data.items()}

    data_sizes = {split: len(dat) for split, dat in data.items()} # for computing accuracy
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn = dataset.collate_fn)
        for split, dataset in datasets.items()
    }

    # Model
    embeddings = torch.load(args.cache_dir / 'embeddings.pt')

    model = SeqTagger(
        embeddings = embeddings,
        hidden_size = args.hidden_size, 
        num_layers = args.num_layers,
        dropout = args.dropout,
        bidirectional = args.bidirectional,
        num_class = len(tag2idx),
        use_crf = args.use_crf,
    ).to(args.device)

    optimizer = Adam(model.parameters(), lr = args.lr)

    # Training
    prec_best = 0.0
    epoch_bar = trange(args.num_epoch, desc='Epoch')
    for epoch in epoch_bar:
        print()
        
        # Training set
        pred, true = train_per_epoch(dataloaders[TRAIN], model, optimizer, args.clip)

        # Validation set
        pred, true = validate_per_epoch(dataloaders[DEV], model)
        #pred, true = train_per_epoch(dataloaders[DEV], model, optimizer, args.clip)

        pred = [[datasets[DEV].idx2label(element) for element in p] for p in pred]  
        true = [[datasets[DEV].idx2label(element) for element in t] for t in true]
        acc, prec = evaluate(pred, true)
        print(f"Validation acc and prec: ({acc}, {prec})")

        # ok = sum([p == t for p, t in zip(pred, true)])
        # prec = ok / len(pred)
        # print(prec)

        # Save the best model
        if prec >  prec_best:
            prec_best = prec
            save_ckpt(args.ckpt_dir, model, optimizer, epoch)

def save_ckpt(ckpt_dir: Path, model, optimizer, epoch):
    ckpt_fp = ckpt_dir / 'model_slot.pt'
    states = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(states, ckpt_fp)
        

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=48)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument('--use_crf', type=bool, default=False)

    # optimizer
    parser.add_argument("--clip", type=float, default=5.)
    parser.add_argument("--lr", type=float, default=5e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)