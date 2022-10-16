import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    
    # One for train, one for eval (use dictionary to index)
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # Note that split_data is List[Dict]
    # key = 'text', 'intent', 'id'

    # TODO: crecate DataLoader for train / dev datasets
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, batch_size=args.batch_size,shuffle=True, collate_fn=split_dataset.collate_fn)
        for split, split_dataset in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings = embeddings,
        hidden_size = args.hidden_size,
        num_layers = args.num_layers,
        bidirectional = args.bidirectional,
        dropout = args.dropout,
        num_class = len(intent2idx),
    ).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    acc_best = 0

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        loss_train, acc_train = train_per_epoch(dataloaders[TRAIN], model, optimizer)
        print()
        print(f'\ttrain_loss = {loss_train}, train_acc = {acc_train}')

        # TODO: Evaluation loop - calculate accuracy and save model weights
        loss_val, acc_val = validate_per_epoch(dataloaders[DEV], model)
        if acc_val > acc_best:
            acc_best = acc_val
            save_ckpt(args.ckpt_dir, model, optimizer, epoch)
        print(f'\tval_loss = {loss_val}, val_acc = {acc_val}')


# return (training loss, accuracy)
def train_per_epoch(dataloader, model, optimizer):
    model.train()

    loss_list = []
    ok, overall = 0, 0
    for batch in dataloader:
        # forward pass
        batch['text'] = torch.tensor(batch['text']).to(args.device)
        batch['intent'] = torch.tensor(batch['intent']).to(args.device)
        output_dict = model(batch)
        loss = output_dict['loss']

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store loss
        loss_list.append(loss.detach())

        # store classification result
        ok += output_dict['ok']
        overall += output_dict['overall']
    
    loss_list = torch.tensor(loss_list)
    return loss_list.mean().item(), (ok / overall)  # (loss, accuracy)

# return (val_loss, val_acc)
@torch.no_grad()
def validate_per_epoch(dataloader, model):
    model.eval()

    ok, overall = 0, 0
    loss_list = []
    for batch in dataloader:
        batch['text'] = torch.tensor(batch['text']).to(args.device)
        batch['intent'] = torch.tensor(batch['intent']).to(args.device)
        output_dict = model(batch)

        loss_list.append(output_dict['loss'])
        ok += output_dict['ok']
        overall += output_dict['overall']

    loss_list = torch.tensor(loss_list)
    return loss_list.mean().item(), ok / overall
    

def save_ckpt(ckpt_dir: Path, model, optimizer, epoch):
    ckpt_path: str = ckpt_dir / 'model_intent.pt'
    states = {
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'epoch': epoch,
    }
    torch.save(states, ckpt_path)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
