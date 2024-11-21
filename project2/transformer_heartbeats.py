import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from biosppy.signals import ecg
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import math

np.random.seed(32)

### Signal dataset
class SignalDataset(Dataset):
    def __init__(self, signals, lengths, labels):
        self.signals = signals
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.lengths[idx], self.labels[idx]


def collate_fn(batch):
    signals, lengths, labels = zip(*batch)
    signals = torch.stack(signals)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)
    signals = signals.to(device)
    lengths = lengths.to(device)
    attention_mask = torch.arange(signals.size(1), device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    return signals, attention_mask, labels


# # Transformer for signal beats
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model) #(seq_len, d_model)
        
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #(seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        
        pe = pe.unsqueeze(0) #(1, seq_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, eps:float = 10**-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True) # We use the last dimension
        std = x.std(dim=-1, keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias

class FeedFwd(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super(FeedFwd, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # (b_size, seq_len, d_model) -> (b_size, seq_len, dff) -> (b_size, seq_len, d_model)
        x = self.relu(self.linear1(x))
        x = self.dropout(self.linear2(x))
        return x
        #return self.linear2(self.dropout(self.linear1(x)))


class MultiheadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float): # h -> num heads
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h # dim of each head
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask != None:
            mask = mask.unsqueeze(1).unsqueeze(2) # (b_size, 1, 1, seq_len) -> make this to match the attention_Score size
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (b_size, h, seq_len, seq_len)
        if dropout != None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores@value), attention_scores
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (b_size, seq_len, d_model) -> (b_size, seq_len, d_model)
        key = self.w_k(k)
        val = self.w_v(v)
        
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # Divide to heads
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        val = val.view(val.shape[0], val.shape[1], self.h, self.d_k).transpose(1,2)
        # Outpus size at this point (b_size, h, seq_len, d_k)
        
        # Apply mask
        x, self.attention_scores = MultiheadAttention.attention(query, key, val, mask, self.dropout)
        
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k) # Go back to (b_size, seq_len, h, d_k) and then (b_size, seq_len, d_model)
        
        return self.w_o(x) # (b_size, seq_len, d_model)

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention, feed_forward, dropout):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
        
        
class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNorm()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, encoder, pos_enc, src_size, d_model, output_dim):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.pos_enc = pos_enc
        self.input_proj = nn.Linear(src_size, d_model)
        self.out_proj = nn.Linear(d_model, output_dim)
        
    def encode(self, src, src_mask):
        src = self.input_proj(src)
        src = self.pos_enc(src)
        encoder_output = self.encoder(src, src_mask)
        # MLP
        pre_out = encoder_output.mean(dim=1) #Avg pooling over seq length -> really helpful?
        logits = self.out_proj(pre_out)
        return logits
        

def build_transformer(src_size, src_seq_len, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048, output_dim=4):
    # Positional encoding
    pe = PositionalEncoding(d_model, src_seq_len, dropout)
    # Encoder
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiheadAttention(d_model, h, dropout)
        feed_fwd = FeedFwd(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_fwd, dropout)
        encoder_blocks.append(encoder_block)
        
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    
    transformer = Transformer(encoder, pe, src_size, d_model, output_dim)
    
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
            
    return transformer


def split_indices(n, val_pct):
    n_val = int(val_pct*n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

if __name__ == "__main__":
    data = pd.read_csv('original_data/train.csv', index_col='id')
    train_y = data['y']
    labels = train_y.to_numpy()

    # See if there is any difference between using the filtered heartbeats or not
    with open('data/heartbeat_templates_ecg.pkl', 'rb') as f:
        heartbeats = pickle.load(f)
        
    # with open('data/heartbeat_filtered_ecg.pkl', 'rb') as f:
    #     heartbeats = pickle.load(f)

    # Longest seq of beats
    max_length_heartbeats = max([len(i) for i in heartbeats])
    beat_length = 180 # All heartbeats have the same length

    # Padded heartbeats
    padded_heartbeats = []
    lengths = []

    for heartbeat in heartbeats:
        length = len(heartbeat)
        lengths.append(length)
        pad = np.zeros((max_length_heartbeats-length, 180))
        padded_heartbeats.append(np.concatenate((heartbeat, pad),axis=0))

    # Transform heartbeats to tensor
    padded_heartbeats = np.array(padded_heartbeats)
    padded_heartbeats = torch.tensor(padded_heartbeats, dtype=torch.float32)
    labels_heartbeats = torch.tensor(labels)
    lengths_heartbeats = torch.tensor(lengths)

    # Train val split
    train_idxs, val_idxs = split_indices(len(padded_heartbeats), 0.2)
    train_x = padded_heartbeats[train_idxs]
    val_y = labels_heartbeats[val_idxs]
    train_y = labels_heartbeats[train_idxs]
    val_x = padded_heartbeats[val_idxs]
    train_lengths = lengths_heartbeats[train_idxs]
    val_lengths = lengths_heartbeats[val_idxs]

    # Let's make a dataset for all the individual beats - 1D conv

    labels_beats = []
    beats = []

    for i, beat in enumerate(heartbeats):
        samples = beat.shape[0]
        beats.append(torch.tensor(beat, dtype=torch.float32))
        labels_beats.append(torch.full((samples,), labels[i]))
        
    beats = torch.cat(beats, dim=0)
    labels_beats = torch.cat(labels_beats)

    # Train - val split for individual beats

    train_idxs_beats, val_idxs_beats = split_indices(len(beats), 0.2)
    train_x_beats = beats[train_idxs_beats]
    val_y_beats = labels_beats[val_idxs_beats]
    train_y_beats = labels_beats[train_idxs_beats]
    val_x_beats = beats[val_idxs_beats]

    # For the moment training without a scaler but try also with scaler later.

    # ### Create dataset

    train_dataset = SignalDataset(train_x, train_lengths, train_y)
    val_dataset = SignalDataset(val_x, val_lengths, val_y)

    BATCH_SIZE = 64

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    d_model = 512
    nhead = 8
    num_encoder_layers=3
    num_decoder_layers = 0
    dim_feedforward = 2048
    dropout = 0.1
    output_dim = 4
    LR = 1e-4
    EPOCHS = 50

    model = build_transformer(src_size=180, src_seq_len=max_length_heartbeats, d_model=d_model, N=num_encoder_layers, h=nhead, dropout=dropout, d_ff=dim_feedforward, output_dim=4)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss=0
        for batch in train_dataloader:
            inputs, attention_mask, targets = batch
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model.encode(inputs, attention_mask)
            loss = loss_fn(outputs, targets)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        val_loss = 0
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                inputs, attention_mask, targets = batch
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                targets = targets.to(device)
                outputs = model.encode(inputs, attention_mask)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
        avg_train_loss = train_loss/len(train_dataloader)
        avg_val_loss = val_loss/len(val_dataloader)
        print(f"Epoch [{epoch+1}/100] | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        with open('train_losses.pkl', 'wb') as f:
            pickle.dump(train_losses, f)
            
        with open('val_losses.pkl', 'wb') as f:
            pickle.dump(val_losses, f)

        torch.save(model.state_dict(), 'firstAttempt')











