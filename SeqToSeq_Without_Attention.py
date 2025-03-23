#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import torch
import torch.nn as nn
import re
from collections import Counter
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence
import os
seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

path = r'/kaggle/input/eng-french/eng_french.csv'
df = pd.read_csv(path, names=['English','French'], header=0)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = text.split()  
    return tokens

def tokenize_text(tokens,token_to_id):
    tokens = [token_to_id.get(token,0) for token in tokens]
    return [1] + tokens + [2]

def tokenize_and_reverse_text(tokens,token_to_id):
    return [token_to_id.get(token,0) for token in (tokens)]
    
english_sentences = df['English'].dropna().apply(preprocess_text)
french_sentences = df['French'].dropna().apply(preprocess_text)
english_vocab = Counter([token for sentence in english_sentences for token in sentence])
french_vocab = Counter([token for sentence in french_sentences for token in sentence])
english_token_to_id = {token: idx + 1 for idx, token in enumerate(english_vocab)}  
french_token_to_id = {token: idx + 3 for idx, token in enumerate(french_vocab)}

english_token_to_id['<PAD>'] = 0
french_token_to_id['<PAD>'] = 0
french_token_to_id['<SOS>'] = 1
french_token_to_id['<EOS>'] = 2
french_id_to_token= {value:key for key,value in french_token_to_id.items()}
english_vocab_size = len(english_token_to_id)
french_vocab_size = len(french_token_to_id)
english_sequences = english_sentences.apply(lambda x: tokenize_and_reverse_text(x, english_token_to_id))
french_sequences = french_sentences.apply(lambda x: tokenize_text(x, french_token_to_id))


# In[ ]:


class SentencesDataset(Dataset):
    def __init__(self,english_sequences,french_sequences):
        self.english_sequences = english_sequences
        self.french_sequences = french_sequences
        assert len(self.english_sequences) == len(self.french_sequences)

    def __len__(self):
        return len(self.english_sequences)

    def __getitem__(self,idx):
        X= self.english_sequences[idx]
        y= self.french_sequences[idx]
        return torch.tensor(X,dtype=torch.long).to(device),torch.tensor(y,dtype=torch.long).to(device)


# In[ ]:


def collate_fn(batch):
    X,y = zip(*batch)
    X_lengths = [len(item) for item in X]
    y_lengths = [len(item) for item in y]
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    return X_padded, y_padded, X_lengths, y_lengths


# In[ ]:


english_temp, french_temp = english_sequences.reset_index(drop=True), french_sequences.reset_index(drop=True)


# In[ ]:


dataset = SentencesDataset(english_temp,french_temp)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,collate_fn = collate_fn)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,collate_fn = collate_fn)


# In[ ]:


EMBEDDING_DIM = 30
HIDDEN_DIM = 128
NUM_LAYERS = 4
DROPOUT = 0.3
SRC_VOCAB_SIZE = english_vocab_size  
PAD_IDX = 0 
TRG_VOCAB_SIZE = french_vocab_size  


# In[ ]:


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout,padding_idx):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, lengths):
        """
        :param src: Source sequence (batch_size, src_len)
        :return: Encoder outputs and hidden states
        """
        embedded = self.dropout(self.embedding(src)) 
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.rnn(packed_input) 
        return hidden, cell
         


# In[ ]:


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.out = nn.Linear(hidden_dim,output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, hidden, cell):
        embedded = self.dropout(self.embedding(target)) 
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        outputs = self.out(outputs.squeeze(1))
        return outputs, hidden, cell


# In[ ]:


class SeqToSeq(nn.Module):
    def __init__(self, encoder, decoder):
        super(SeqToSeq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, src_lengths, teacher_forcing_ratio=0.5):
        """
        :param src: Source sequence (batch_size, src_len)
        :param trg: Target sequence (batch_size, trg_len)
        :param src_lengths: Lengths of the source sequences
        :param trg_lengths: Lengths of the target sequences
        :param teacher_forcing_ratio: Probability of using teacher forcing
        :return: Decoder outputs (batch_size, trg_len, output_dim)
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.out.out_features

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, output_dim)

        # Encode the source sequence
        hidden, cell = self.encoder(src, src_lengths)

        # First input to the decoder is the <sos> token
        input = trg[:, 0].unsqueeze(1)  # (batch_size, 1)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden[-NUM_LAYERS:, :, :], cell[-NUM_LAYERS:, :, :])  # Decoder forward pass
            outputs[:, t, :] = output  # Store the output

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)  # Get the predicted next token

            input = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs


# In[ ]:


encoder = Encoder(
    input_dim=SRC_VOCAB_SIZE,
    emb_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    padding_idx = 0
)


# In[ ]:


decoder = Decoder(
    output_dim=TRG_VOCAB_SIZE,
    emb_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)


# In[ ]:


model = SeqToSeq(encoder, decoder).to(device)
if os.path.exists("/kaggle/input/pretrained_eng_fren_seqseq/pytorch/default/1/seq2seq_model_weights.pth"):
    print("Predefined model exists")
    model.load_state_dict(torch.load("/kaggle/input/pretrained_eng_fren_seqseq/pytorch/default/1/seq2seq_model_weights.pth"))


# In[ ]:


EPOCHS = 10
LEARNING_RATE = 0.01
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# In[ ]:


def train():
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for src, trg, src_lengths,_ in train_loader:
            optimizer.zero_grad()
            output = model(src, trg, src_lengths).to(device)
            output = output[:, 1:].reshape(-1, output.shape[-1])  
            trg = trg[:, 1:].reshape(-1) 
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        torch.save(model.state_dict(), "seq2seq_model_weights.pth")
        print(f"Epoch: {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader):.4f}")


# In[ ]:


train()


# In[ ]:


def eval():
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, trg, src_lengths, _ in val_loader:
                
            output = model(src, trg, src_lengths, teacher_forcing_ratio=0.0)
            output = output[:, 1:].reshape(-1, output.shape[-1]).to(device)  # Ignore <sos> token
            trg = trg[:, 1:].reshape(-1)
    
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
        print(epoch_loss / len(val_loader))


# In[ ]:


eval()


# In[ ]:


def infer(model, src, src_lengths, trg_vocab, max_len=50):
    
    model.eval()
    with torch.no_grad():
        hidden, cell = model.encoder(src, src_lengths)
        # Start with <sos> token
        trg_vocab_size = model.decoder.out.out_features
        input = torch.tensor([[1]], device=device)  # (1, 1)
        predictions = []

        for _ in range(max_len):
            output, hidden, cell = model.decoder(input, hidden, cell)
            
            top1 = output.argmax(1)  
            predictions.append(top1.item())
            if top1.item() == trg_vocab['<EOS>']:
                break

            input = top1.unsqueeze(1).to(device)  
    return [french_id_to_token[idx] for idx in predictions]


# In[ ]:


sentence = "I like you"
sentence = preprocess_text(sentence)
sentence = tokenize_and_reverse_text(sentence, english_token_to_id)
output = infer(model, torch.tensor([sentence]).to(device),[len(sentence)],french_token_to_id)


# In[ ]:


print(output)

