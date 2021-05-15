import numpy as np
import torch
import transformers
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, imgs, seqs, outs):
        self.imgs = np.expand_dims(imgs, axis=1)
        self.outs = outs
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        self._setup_seq_ids(seqs)

    def _setup_seq_ids(self, seqs):
        self.seq_ids = []
        for seq in seqs:
            self.seq_ids.append(torch.Tensor(self.tokenizer.encode(seq)).view(1, -1) \
                    .type(torch.LongTensor))
        self.seq_ids = torch.vstack(self.seq_ids)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        output = np.zeros((2))
        output[self.outs[idx]] = 1

        return self.seq_ids[idx], self.imgs[idx].astype(np.double), output
