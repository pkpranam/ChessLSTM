import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, embed_dropout, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.embed_ln = nn.LayerNorm(embed_dim)
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # attention stuff
        self.attn_W = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_ln = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.pad_idx = pad_idx
    
    def forward(self, x, lengths):
        # embeddings
        emb = self.embed(x)
        emb = self.embed_dropout(emb)
        emb = self.embed_ln(emb)
        
        # lstm
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        # attention mechanism
        score = self.attn_v(torch.tanh(self.attn_W(out)))
        
        # mask padding
        mask = (x != self.pad_idx).unsqueeze(-1)
        score[~mask] = -1e4
        
        attn_weights = torch.softmax(score, dim=1)
        context = (attn_weights * out).sum(dim=1)
        
        # final prediction
        context = self.dropout(context)
        logits = self.fc(context)
        return logits
