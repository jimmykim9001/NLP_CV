from tqdm import tqdm
import IPython
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import transformers
import pytorch_lightning as pl

class TextConvolution(nn.Module):
    def __init__(self, filter_size=3, input_channels=1, output_channels=1):
        super().__init__()
        # the number of input/output channels of the text-adjusted convolution
        self.input_channels, self.output_channels = input_channels, output_channels
        self.filter_size = filter_size

        # out of the box bert model
        self.bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')
        # ignore gradient updates for bert
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.hidden_size = 768

        # attention layer that calculates which tokens in the text input are important
        # for each combination of input-to-output channel
        self.alpha_lin = nn.Linear(self.hidden_size, input_channels * output_channels)
        self.softmax = nn.Softmax(dim=1) # softmax to normalize all alpha across sequence

        # linear layer that is responsible for converting BERT embeddings into
        # a convolutional filter
        self.conv_linear = nn.Linear(self.hidden_size, self.filter_size ** 2)

    def forward(self, input_tensor, input_image):
        _, input_channels, input_height, input_width = input_image.shape
        batch_size, seq_len = input_tensor.shape

        temp = self.bert_model(input_tensor)
        #print(type(temp))
        #print(temp)
        last_bert = temp.last_hidden_state
        #last_bert = self.bert_model(input_tensor).last_hidden_state
        reshaped_last_bert = last_bert.view(batch_size, self.hidden_size, seq_len)

        alpha = self.alpha_lin(last_bert)
        norm_alpha = self.softmax(alpha) # [batch_size, seq_len, input_channels * output_channels]

        # utilize attention to create hidden state representations. This then
        # is converted to convolutions
        weights = torch.matmul(reshaped_last_bert, norm_alpha) # [batch_size, hidden_size, input * output]
        weights = weights.view(batch_size, -1, self.hidden_size)
        weights = weights.view(batch_size, self.output_channels, self.input_channels, self.hidden_size)
        weights = self.conv_linear(weights) 
        weights = weights.view(batch_size, self.output_channels, self.input_channels, self.filter_size, self.filter_size)

        # apply ith-convolution to ith image in batch
        adj_conv = F.conv2d(\
                input_image.view(1, batch_size * input_channels, input_height, input_width),
                weights.view(batch_size * self.output_channels, self.input_channels, self.filter_size, self.filter_size),
                groups=batch_size, padding=(self.filter_size - 1) // 2
                ) # [batch_size, output_channels, size - f + 1, size - f + 1]
        adj_conv = adj_conv.view(batch_size, self.output_channels, adj_conv.shape[-2], adj_conv.shape[-1])
        adj_conv = F.relu(adj_conv)
        return adj_conv

class BERTModelLightning(pl.LightningModule):
    """
    [CLS, find, the, person] (attention) -> [768] -> convolution matrix -> output
    conv_filters must include input
    """
    def __init__(self, conv_filters, filter_size=3, output_channels=1, last_size=16, maxpool_skip=2):
        super().__init__()
        self.relu_fn = nn.ReLU()
        self.maxpool_skip = maxpool_skip
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.conv_filters = conv_filters
        if self.conv_filters:
            self.conv_list = [nn.Conv2d(conv_filters[idx], conv_filters[idx + 1], 3, padding=1) \
                    for idx in range(len(conv_filters) - 1)]

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.tc = TextConvolution(filter_size, self.conv_filters[-1], output_channels)
        self.final_linear = nn.Linear(output_channels * (last_size ** 2), 2)
        self.pred_softmax = nn.Softmax(dim=1)
        """
        #obtaining stuff
        self.zeroProb_tr = []
        self.oneProb_tr = []
        self.preds_tr= []
        self.corrLabels_tr=[]
        self.zeroProb_va = []
        self.oneProb_va = []
        self.preds_va= []
        self.corrLabels_va=[]
        """
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), 0.01)
        return optimizer

    def forward(self, input_tensor, input_image):
        for idx in range(len(self.conv_filters) - 1):
            input_image = self.conv_list[idx](input_image)
            input_image = self.relu_fn(input_image)

            # max pooling layers (don't include at the end)
            if idx % self.maxpool_skip == self.maxpool_skip - 1 and idx != len(self.conv_filters) - 2:
                input_image = self.maxpool(input_image)

        adj_conv = self.tc(input_tensor, input_image)

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
        self.log('train_accuracy', int(correct), on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss

    def validation_step(self, train_batch, batch_idx, dataloader_nb):
        seq_ids, imgs, outs= train_batch
        #seq_ids, imgs, outs
        print(dataloader_nb)
        seq_ids = torch.unsqueeze(seq_ids, 0)
        
        imgs = torch.unsqueeze(torch.Tensor(imgs), 0)
        outs = torch.unsqueeze(torch.Tensor(outs), 0)
        output = self(seq_ids, imgs)
        loss = self.loss_fn(output, outs)
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        correct = (outs.argmax() == self.pred_softmax(output).argmax()).item()
        self.log('valid_accuracy', int(correct), on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss
