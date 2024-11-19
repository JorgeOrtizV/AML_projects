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

np.random.seed(32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(DynamicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0))/ d_model)).to(x.device)
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        return self.dropout(x+pe)
    
class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, output_dim):
        super(Transformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        #self.positional_encoding = PositionalEncoding(d_model)
        self.positional_encoding = DynamicPositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Enables input shape (batch, seq, feature)
        )
        self.output_fc = nn.Linear(d_model, output_dim)

    def forward(self, src, src_key_padding_mask=None):
        # Apply input embedding
        src = self.input_fc(src)
        src = self.positional_encoding(src)
        # Pass through Transformer Encoder
        output = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
        # Mean pooling
        pooled_output = output.mean(dim=1)
        # Generate final output
        return self.output_fc(pooled_output)
    
class SignalDataset(Dataset):
    def __init__(self, signals, lengths, labels):
        self.signals = signals
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.lengths[idx], self.labels[idx]

def split_indices(n, val_pct):
    n_val = int(val_pct*n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

def collate_fn(batch):
    signals, lengths, labels = zip(*batch)
    signals = torch.stack(signals)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)
    attention_mask = torch.arange(signals.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
    return signals, attention_mask, labels

if __name__ == "__main__":
    data = pd.read_csv('original_data/train.csv', index_col='id')
    train_y = data['y']

    with open('data/filtered_ecg.pkl', 'rb') as f:
        filtered_signals = pickle.load(f)
        
    with open('data/heartbeat_templates_ecg.pkl', 'rb') as f:
        heartbeats = pickle.load(f)

    train_idxs, val_idxs = split_indices(len(filtered_signals), 0.2)

    max_length = max([len(i) for i in filtered_signals])
    padded_signals = []
    lengths = []

    for signal in filtered_signals:
        length = len(signal)
        lengths.append(length)
        # 0 padding up to max_length
        padded_signal = np.pad(signal, (0, max_length-length), 'constant')
        padded_signals.append(padded_signal)

    # Tensors
    padded_signals = np.array(padded_signals)
    padded_signals = torch.tensor(padded_signals, dtype=torch.float32)
    labels = torch.tensor(train_y.to_numpy())
    lengths = torch.tensor(np.array(lengths))

    # Train val split
    train_x = padded_signals[train_idxs]
    train_y = labels[train_idxs]
    val_x = padded_signals[val_idxs]
    val_y = labels[val_idxs]
    train_lengths = lengths[train_idxs]
    val_lengths = lengths[val_idxs]

    train_dataset = SignalDataset(train_x, train_lengths, train_y)
    val_dataset = SignalDataset(val_x, val_lengths, val_y)

    # Hyperparams
    input_dim = 1
    d_model = 64
    nhead = 8
    num_encoder_layers=3
    num_decoder_layers = 0
    dim_feedforward = 256
    dropout = 0.1
    output_dim = 4
    LR = 1e-4
    EPOCHS = 50
    BATCH_SIZE = 64

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = Transformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_dim=output_dim
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            inputs, attention_mask, targets = batch
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(-1)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, src_key_padding_mask=attention_mask)
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
                inputs = inputs.unsqueeze(1) # (batch_size, seq_len, 1)
                attention_mask = attention_mask.to(device)
                targets = targets.to(device)
                outputs = model(inputs, src_key_padding_mask=attention_mask)
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

    