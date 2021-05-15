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
import pandas as pd
from models import BERTModelLightning

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
            #top
                input_npy[:halved,:] = np.random.randint(2, size=((halved, dimension)))
            true = 0
        else:
            if isLeft:
                input_npy[:,halved:] = np.random.randint(2, size=((dimension,halved)))
            else:
                input_npy[halved:,:] = np.random.randint(2, size=((halved,dimension)))
            #bottom
            true = 1
        for idx_d, direction in enumerate(names):
            images.append(input_npy)
            sequences.append(direction)
            outputs.append(int(idx_d == true))
    return np.stack(images, axis=0), np.array(sequences), np.array(outputs)

class TestAdjConv(unittest.TestCase):
    def test_basic(self):
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        bbm = BERTModelLightning([1, 16, 32, 32, 64], output_channels=8)
        imgs, seqs, outs = generate_dataset(100, False)
        valiimgs, valiseqs, valiouts = generate_dataset(50, False)
        # process seq_ids
        seq_ids = []
        for seq in seqs:
            seq_ids.append(torch.Tensor(tokenizer.encode(seq)).view(1, -1).type(torch.LongTensor))
        seq_ids = torch.vstack(seq_ids)

        val_seq_ids = []
        for seq in valiseqs:
            val_seq_ids.append(torch.Tensor(tokenizer.encode(seq)).view(1, -1).type(torch.LongTensor))
        val_seq_ids = torch.vstack(val_seq_ids)
        # process imgs
        imgs = np.expand_dims(imgs, axis=1)
        valiimgs = np.expand_dims(valiimgs, axis=1)
        # process outputs
        final_outs = np.zeros((outs.shape[0], 2))
        final_outs[np.arange(outs.shape[0]), outs] = 1
        outs = final_outs

        val_final_outs = np.zeros((valiouts.shape[0], 2))
        val_final_outs[np.arange(valiouts.shape[0]), valiouts] = 1
        valiouts = val_final_outs

        tb_logger = pl_loggers.CSVLogger('logs/')
        trainer = pl.Trainer(max_epochs=10, logger=tb_logger, log_every_n_steps=1)

        idx = np.random.permutation(imgs.shape[0])
        idx_val = np.random.permutation(valiimgs.shape[0])
        imgs, seq_ids, outs = imgs[idx], seq_ids[idx], outs[idx]
        valiimgs, val_seq_ids, valiouts = valiimgs[idx_val], val_seq_ids[idx_val], valiouts[idx_val]
        train_dataloader =(seq_ids, imgs, outs)
        val_dataloader=(val_seq_ids, valiimgs, valiouts)
        #, val_dataloader
        trainer.fit(bbm, train_dataloader)
        #pass in data to model here and use that output for csv
        #modelData = [bbm.zeroProb_tr, bbm.oneProb_tr, bbm.preds_tr, bbm.corrLabels_tr]
        #seq_ids = torch.unsqueeze(seq_ids, 0)
        imgs = torch.Tensor(imgs)
        outs = torch.Tensor(outs)
        probs = bbm(seq_ids, imgs)
        tostore= bbm.pred_softmax(probs).tolist()
        zeros, ones =map(list, zip(*tostore))
        corr =outs.argmax(dim=1)
        d= {'prob_zero': zeros, 'prob_one': ones, 'correct': corr}
        df = pd.DataFrame(data=d)
        df.to_csv('logs/res2.csv')


if __name__ == "__main__":
    unittest.main()
