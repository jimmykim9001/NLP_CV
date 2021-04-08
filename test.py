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

import adj_conv

def generate_dataset(n=100):
    outputs, images, sequences = [], [], []
    for idx in range(n):
        input_npy = np.zeros((5, 5))
        if random.random() < 0.5:
            input_npy[:,:2] = np.random.randint(2, size=((5,2)))
            true = 0
        else:
            input_npy[:,3:] = np.random.randint(2, size=((5,2)))
            true = 1
        for idx_d, direction in enumerate(['left', 'right']):
            images.append(input_npy)
            sequences.append(direction)
            outputs.append(int(idx_d == true))
    return np.stack(images, axis=0), np.array(sequences), np.array(outputs)

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.lin1 = nn.Linear(25, 9)
        self.nlp_conv1 = adj_conv.NLPConv(1, 1, 3)

    def forward(self, x):
        flattened_input = torch.flatten(x)
        weights = self.lin1(flattened_input)
        weights = weights.view(1, 1, 3, 3)
        output = self.nlp_conv1(weights, x)
        return output

class BasicBERTModel(pl.LightningModule):
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        # the number of input/output channels of the text-adjusted convolution
        self.input_channels, self.output_channels = input_channels, output_channels

        # out of the box bert model
        self.bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = 768

        # attention layer that calculates which tokens in the text input are important
        # for each combination of input-to-output channel
        self.alpha_lin = nn.Linear(self.hidden_size, input_channels * output_channels)
        self.softmax = nn.Softmax(dim=1) # softmax to normalize all alpha across sequence

        # linear layer that is responsible for converting BERT embeddings into
        # a convolutional filter (3x3)
        self.conv_linear = nn.Linear(self.hidden_size, 9)

        self.final_linear = nn.Linear(output_channels * 9, 2) # hard-coded for now

        self.pred_softmax = nn.Softmax(dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), 0.01)
        return optimizer

    def forward(self, input_tensor, input_image):
        _, input_channels, input_height, input_width = input_image.shape
        batch_size, seq_len = input_tensor.shape

        last_bert = self.bert_model(input_tensor).last_hidden_state
        reshaped_last_bert = last_bert.view(batch_size, self.hidden_size, seq_len)

        alpha = self.alpha_lin(last_bert)
        norm_alpha = self.softmax(alpha) # [batch_size, seq_len, 1]

        # utilize attention to create hidden state representations. This then
        # is converted to convolutions
        weights = torch.matmul(reshaped_last_bert, norm_alpha)
        weights = weights.view(batch_size, -1, self.hidden_size)
        weights = weights.view(batch_size, self.output_channels, self.input_channels, self.hidden_size)
        weights = self.conv_linear(weights) 
        weights = weights.view(batch_size, self.output_channels, self.input_channels, 3, 3)

        # apply ith-convolution to ith image in batch
        adj_conv = F.conv2d(\
                input_image.view(1, batch_size * input_channels, input_height, input_width),
                weights.view(batch_size * self.output_channels, self.input_channels, 3, 3),
                groups=batch_size
                ) # [batch_size, output_channels, size - f + 1, size - f + 1]
        adj_conv = adj_conv.view(batch_size, self.output_channels, adj_conv.shape[-2], adj_conv.shape[-1])
        adj_conv = F.relu(adj_conv)

        flattened = torch.flatten(adj_conv, start_dim=1)
        output = self.final_linear(flattened)
        return output

    def training_step(self, train_batch, batch_idx):
        seq_ids, imgs, outs = train_batch
        seq_ids = torch.unsqueeze(seq_ids, 0)
        imgs = torch.unsqueeze(torch.Tensor(imgs), 0)
        outs = torch.unsqueeze(torch.Tensor(outs), 0)
        output = self(seq_ids, imgs)
        loss = self.loss_fn(output, outs)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        correct = (outs.argmax() == self.pred_softmax(output).argmax()).item()
        self.log('accuracy', int(correct), on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss

class TestAdjConv(unittest.TestCase):
    def test_basic(self):
        bm = BasicModel()
        input_tensor = torch.rand(1, 1, 5, 5)
        output_tensor = bm(input_tensor)

        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        bbm = BasicBERTModel(output_channels=16)
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
        trainer = pl.Trainer(max_epochs=5)

        idx = np.random.permutation(imgs.shape[0])
        imgs, seq_ids, outs = imgs[idx], seq_ids[idx], outs[idx]

        trainer.fit(bbm, (seq_ids, imgs, outs))

if __name__ == "__main__":
    unittest.main()
