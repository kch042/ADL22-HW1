import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    # TODO: implement main function
    with open(args.cache_dir / 'vocab.pkl', 'rb') as f:
        vocab: Vocab = pickle.load(f)
    tag2idx = json.loads((args.cache_dir / 'tag2idx.json').read_text())

    data = json.loads((args.data_dir / 'test.json').read_text())
    dataset = SeqTaggingClsDataset(
        data = data,
        vocab = vocab,
        label_mapping = tag2idx,
        max_len = args.max_len
    )
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, collate_fn = dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / 'embeddings.pt')
    model = SeqTagger(
      embeddings = embeddings,
      hidden_size = args.hidden_size, 
      num_layers = args.num_layers,
      dropout = args.dropout, 
      bidirectional = args.bidirectional,
      num_class = len(tag2idx),
    ).to(args.device)
    
    ckpt = torch.load(args.ckpt_dir / 'model_slot.pt')
    model.load_state_dict(ckpt['model'])
    
    model.eval()
    model.testing = True

    with open(args.pred_file, 'w') as f:
        f.write('id,tags\n')
        
        for batch in dataloader:
            batch['tokens'] = torch.tensor(batch['tokens']).to(args.device)
            output_dict = model(batch)

            for i, length, tags in zip(batch['id'], batch['seq_len'], output_dict['pred']):
                f.write(f'{i},')
                pred = tags.tolist()
                for i in range(length-1):
                    f.write(f'{dataset.idx2label(pred[i])} ')
                f.write(f'{dataset.idx2label(pred[length-1])}\n')
                




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
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)