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

def predict(bbm, dataset, save_path='logs/train_results.csv'):
    imgs, outs, seq_ids = torch.Tensor(dataset.imgs), torch.Tensor(dataset.outs), dataset.seq_ids

    probs = bbm(seq_ids, imgs)
    tostore= bbm.pred_softmax(probs).tolist()
    zeros, ones = map(list, zip(*tostore))
    df = pd.DataFrame({'prob_zero': zeros, 'prob_one': ones, 'correct': outs})
    df.to_csv(save_path)

class TestAdjConv(unittest.TestCase):
    def test_basic(self):
        bbm = BERTModelLightning([1, 16, 32, 32, 64], output_channels=8)
        train_ds = ToyDataset(*generate_dataset())
        valid_ds = ToyDataset(*generate_dataset(50))

        train_dl = DataLoader(train_ds, batch_size=4)
        valid_dl = DataLoader(valid_ds, batch_size=4)

        tb_logger = pl_loggers.CSVLogger('logs/')
        trainer = pl.Trainer(max_epochs=1, logger=tb_logger, log_every_n_steps=1)

        trainer.fit(bbm, train_dl, valid_dl)
        predict(bbm, train_ds)
        # predict(bbm, train_ds)

if __name__ == "__main__":
    unittest.main()
