import unittest
import random

from tqdm import tqdm
import IPython
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from models import BERTModelLightning
from dataset import ToyDataset


def generate_dataset(n=100, isLeft=True, dimension=32):
    outputs, images, sequences = [], [], []
    halved = int(dimension/2)
    if isLeft:
        names= ['left', 'right']
    else:
        names = ['top', 'bottom']
    for idx in range(n):
        input_npy = np.zeros((dimension, dimension))
        if random.random() < 0.5:
            if isLeft:
                input_npy[:,:halved] = np.random.randint(2, size=((dimension,halved)))
            else:
                input_npy[:halved,:] = np.random.randint(2, size=((halved, dimension)))
            true = 0
        else:
            if isLeft:
                input_npy[:,halved:] = np.random.randint(2, size=((dimension,halved)))
            else:
                input_npy[halved:,:] = np.random.randint(2, size=((halved,dimension)))
            true = 1
        for idx_d, direction in enumerate(names):
            images.append(input_npy)
            sequences.append(direction)
            outputs.append(int(idx_d == true))
    return np.stack(images, axis=0), np.array(sequences), np.array(outputs)

def predict(bbm, dataloader, save_path='logs/train_results.csv', device='cpu'):
    bbm = bbm.to(device)
    all_zeros, all_ones, all_outs = [], [], []
    for seq_ids, imgs, outs in dataloader: 

        probs = bbm(seq_ids.to(device), imgs.to(device))
        tostore= bbm.pred_softmax(probs).tolist()
        zeros, ones = map(list, zip(*tostore))

        # save
        all_zeros += zeros
        all_ones += ones
        all_outs += outs.argmax(dim=1).tolist()
    df = pd.DataFrame({'prob_zero': all_zeros, 'prob_one': all_ones, 'correct': all_outs})
    df.to_csv(save_path)

class TestAdjConv(unittest.TestCase):
    def test_basic(self):
        dims = [16, 32, 64, 128]
        # ns = [100, 200, 300, 400]
        ns = [400, 400, 400, 400]

        for dim, n in zip(dims, ns):
            bbm = BERTModelLightning([1, 16, 32, 64, 128], output_channels=64, last_size=dim//2, device='cuda')
            bbm.to('cuda')
            train_ds = ToyDataset(*generate_dataset(n, dimension=dim))
            valid_ds = ToyDataset(*generate_dataset(n, dimension=dim))

            train_dl = DataLoader(train_ds, batch_size=16)
            valid_dl = DataLoader(valid_ds, batch_size=16)

            tb_logger = pl_loggers.CSVLogger('logs/', name=f'dim{dim}')
            trainer = pl.Trainer(max_epochs=10, logger=tb_logger, log_every_n_steps=1, gpus=1)

            trainer.fit(bbm, train_dl, valid_dl)

            predict(bbm, train_dl, save_path=f'logs/dim{dim}/train_results.csv', device='cuda')
            predict(bbm, valid_dl, save_path=f'logs/dim{dim}/valid_results.csv', device='cuda')

        # train on top/bottom
        # test_ds = ToyDataset(*generate_dataset(isLeft=False, n=10))
        # test_dl = DataLoader(test_ds, batch_size=1)
        #
        # test_ds2 = ToyDataset(*generate_dataset(isLeft=False, n=50))
        # test_dl2 = DataLoader(test_ds2, batch_size=1)
        #
        # test_logger = pl_loggers.CSVLogger('test_logs/')
        # test_trainer = pl.Trainer(max_epochs=5, logger=test_logger, log_every_n_steps=1)
        # test_trainer.fit(bbm, test_dl, test_dl2)
        # predict(bbm, test_ds, save_path='test_logs/test_results.csv')

if __name__ == "__main__":
    unittest.main()
