import unittest
import random

from tqdm import tqdm
import IPython
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import transformers
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from models import BERTModelLightning

def generate_dataset(n=100):
    outputs, images, sequences = [], [], []
    for idx in range(n):
        input_npy = np.zeros((32, 32))
        if random.random() < 0.5:
            input_npy[:,:16] = np.random.randint(2, size=((32,16)))
            true = 0
        else:
            input_npy[:,16:] = np.random.randint(2, size=((32,16)))
            true = 1
        for idx_d, direction in enumerate(['left', 'right']):
            images.append(input_npy)
            sequences.append(direction)
            outputs.append(int(idx_d == true))
    return np.stack(images, axis=0), np.array(sequences), np.array(outputs)

class TestAdjConv(unittest.TestCase):
    def test_basic(self):
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        bbm = BERTModelLightning([1, 16, 32, 32, 64], output_channels=8)
        imgs, seqs, outs = generate_dataset()

        # process seq_ids
        seq_ids = []
        for seq in seqs:
            seq_ids.append(torch.Tensor(tokenizer.encode(seq)).view(1, -1).type(torch.LongTensor))
        seq_ids = torch.vstack(seq_ids)

        # process imgs
        imgs = np.expand_dims(imgs, axis=1)

        # process outputs
        final_outs = np.zeros((outs.shape[0], 2))
        final_outs[np.arange(outs.shape[0]), outs] = 1
        outs = final_outs

        tb_logger = pl_loggers.TensorBoardLogger('logs/')
        trainer = pl.Trainer(max_epochs=10, logger=tb_logger, log_every_n_steps=1)

        idx = np.random.permutation(imgs.shape[0])
        imgs, seq_ids, outs = imgs[idx], seq_ids[idx], outs[idx]

        trainer.fit(bbm, (seq_ids, imgs, outs))

if __name__ == "__main__":
    unittest.main()
