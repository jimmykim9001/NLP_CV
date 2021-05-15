import unittest
import random

from tqdm import tqdm
import IPython
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pandas as pd

from models import BERTModelLightning
from dataset import ToyDataset

def generate_dataset(n=100, dim=32, direction='vertical'):
    # slice objects
    full_slice = slice(None, None) # full dataset
    start_slice = slice(None, dim // 2)
    end_slice = slice(dim // 2, None)

    if direction == 'vertical':
        size = ((dim // 2, dim))
        directions = ['top', 'bottom']
    else:
        size = ((dim, dim // 2))
        directions = ['left', 'right']

    outputs, images, sequences = [], [], []
    for idx in range(n):
        input_npy = np.zeros((dim, dim))
        if random.random() < 0.5:
            use_slice = start_slice
            true = 0
        else:
            use_slice = end_slice
            true = 1

        # create slice object
        if direction == 'vertical':
            curr_slice = use_slice, full_slice
        else:
            curr_slice = full_slice, use_slice

        input_npy[curr_slice] = np.random.randint(2, size=size)

        # iterate over directions
        for idx_d, curr_dir in enumerate(directions):
            images.append(input_npy)
            sequences.append(curr_dir)
            outputs.append(int(idx_d == true))
    return np.stack(images, axis=0), np.array(sequences), np.array(outputs)

class TestAdjConv(unittest.TestCase):
    def test_basic(self):
        bbm = BERTModelLightning([1, 16, 32, 32, 64], output_channels=8)
        train_ds = ToyDataset(*generate_dataset())
        valid_ds = ToyDataset(*generate_dataset(50))

        train_dl = DataLoader(train_ds, batch_size=4)
        valid_dl = DataLoader(valid_ds, batch_size=4)

        tb_logger = pl_loggers.CSVLogger('logs/')
        trainer = pl.Trainer(max_epochs=10, logger=tb_logger, log_every_n_steps=1)

        #, val_dataloader
        trainer.fit(bbm, train_dl, valid_dl)
        modelData = [bbm.zeroProb_tr, bbm.oneProb_tr, bbm.preds_tr, bbm.corrLabels_tr]
        d= {'prob_zero': modelData[0], 'prob_one': modelData[1], 'pred':modelData[2], 'correct': modelData[3]}
        df = pd.DataFrame(data=d)


if __name__ == "__main__":
    unittest.main()
