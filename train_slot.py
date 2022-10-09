import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def train_per_epoch(dataloader, model, optimizer) -> Tuple[float, int]:
    model.train()

    loss_list = []
    correct = 0

    for batch in dataloader:
        # forward pass
        batch['tokens'] = torch.tensor(batch['tokens']).to(args.device)
        batch['tags'] = torch.tensor(batch['tags']).to(args.device)
        output_dict = model(batch)
        
        # store result
        loss = output_dict['loss']
        loss_list.append(loss.detach())
        correct += output_dict['correct']

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_mean = torch.tensor(loss_list).mean().item()
    return loss_mean, correct

@torch.no_grad()
def validate_per_epoch(dataloader, model) -> Tuple[float, int]:
    model.eval()

    loss_list = []
    correct = 0
    for batch in dataloader:
        batch['tokens'] = torch.tensor(batch['tokens']).to(args.device)
        batch['tags'] = torch.tensor(batch['tags']).to(args.device)
        output_dict = model(batch)

        loss = output_dict['loss']
        loss_list.append(loss)
        correct += output_dict['correct']      

    loss_mean = torch.tensor(loss_list).mean().item()
    return loss_mean, correct

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
        num_class = len(tag2idx)
    ).to(args.device)

    optimizer = Adam(model.parameters(), lr = args.lr)

    # Training
    acc_best = 0.0
    epoch_bar = trange(args.num_epoch, desc='Epoch')
    for epoch in epoch_bar:
        # Training set
        loss_train, correct_train = train_per_epoch(dataloaders[TRAIN], model, optimizer)
        acc_train = correct_train / data_sizes[TRAIN]

        print()
        print(f'\ttrain_loss = {loss_train}, train_acc = {acc_train}')

        # Validation set
        loss_val, correct_val = validate_per_epoch(dataloaders[DEV], model)
        acc_val = correct_val / data_sizes[DEV]

        print(f'\tval_loss = {loss_val}, val_acc = {acc_val}')

        # Save the best model
        if acc_val > acc_best:
            acc_val = acc_best
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

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)

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