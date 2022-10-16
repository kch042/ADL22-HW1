# Training

## Environment
Google Colab

### Python version
The current colab default python version is 3.7.
To update python version to 3.9, run
```
!wget -O mini.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
!chmod +x mini.sh
!bash ./mini.sh -b -f -p /usr/local
!conda install -q -y jupyter
!conda install -q -y google-colab -c conda-forge
!python -m ipykernel install --name "py39" --user
```

## Requirement Modules
`!pip install -r requirement.in`

## Preprocessing
`!bash preprocess.sh`

## Training
### Intent Classification
`!python train_intent.py --data_dir <data_dir> --cache_dir <cache_dir> --ckpt_dir <ckpt_dir> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --lr <lr> --batch_size <batch_size> --num_epoch <num_epoch>`

- data_dir: Directory to the train.json and eval.json file.
- cache_dir: Directory to the preprocessed caches.
- ckpt_dir: Directory to save the model file.
- hidden_size: LSTM output hidden state dimension. [default: 512]
- num_layers: number of layers in LSTM [default: 2]
- dropout: dropout rate in the neural networks. [default: 0.2]
- bidirectional: use bidirectional LSTM or not. (if not, unidirectional LSTM is used) [default: True]
- lr: learning rate. [default: 1e-3]
- batch_size: size of data in each mini batch. [default: 128]
- num_epoch: number of epochs for training. [default: 50]


### Slot Tagging
`!python train_slot.py --data_dir <data_dir> --ckpt_dir <ckpt_dir> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --lr <lr> --batch_size <batch_size> --num_epoch <num_epoch> --use_crf <use_crf> --clip <clip>`

- data_dir: Directory to the train.json and eval.json file.
- ckpt_dir: Directory to save the model file.
- hidden_size: LSTM output hidden state dimension. [default: 512]
- num_layers: number of layers in LSTM [default: 2]
- dropout: dropout rate in the neural networks. [default: 0.3]
- bidirectional: use bidirectional LSTM or not. (if not, unidirectional LSTM is used) [default: True]
- lr: learning rate. [default: 5e-4]
- batch_size: size of data in each mini batch. [default: 128]
- num_epoch: number of epochs for training. [default: 50]
- device: train the model on cpu or cuda [default: cuda]
- use_crf: enable use of CRF as final layer. (if not, then a fully connected linear layer + cross entropy loss is used) [default: False]
- clip: gradient clipping. [default: 5.0]

## Prediction
### Intent Classification
`!python test_intent.py --test_file <test_file> --ckpt_path <ckpt_path> --pred_file <pred_file>`

- test_file: Path to the test json file.
- ckpt_path: Path to model checkpoint file.
- pred_file: Path to save the prediction file.

### Slot Tagging
`!python test_slot.py --test_file <test_file> --ckpt_path <ckpt_path> --pred_file <pred_file>

- test_file: Path to the test json file.
- ckpt_path: Path to model checkpoint file.
- pred_file: Path to save the prediction file.