import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import chess
from collections import Counter

# global vocab dicts - will be set by load_vocab
stoi = {}
itos = {}

def load_vocab(dataset_lines):
    """Build vocab from dataset"""
    tokenized = [line.split(", ") for line in dataset_lines]
    all_tokens = [x for sublist in tokenized for x in sublist]
    
    counter = Counter(all_tokens)
    tokens = ["<pad>", "<unk>"] + sorted(counter.keys())
    
    global stoi, itos
    stoi = {tok: i for i, tok in enumerate(tokens)}
    itos = {i: tok for tok, i in stoi.items()}
    
    return len(stoi)

def encode(seq):
    """Convert move sequence to token IDs"""
    return torch.tensor([stoi.get(tok, stoi["<unk>"]) for tok in seq], dtype=torch.long)

def legal_moves_from_sequence(moves):
    """Get legal moves mask for a position"""
    board = chess.Board()
    for move in moves:
        try:
            board.push(board.parse_san(move))
        except ValueError:
            pass
    
    mask = torch.full((len(stoi),), float('-inf'))
    for move in board.legal_moves:
        san = board.san(move)
        if san in stoi:
            mask[stoi[san]] = 0.0
    return mask

class ChessDataset(Dataset):
    def __init__(self, tokenized_inputs, output_moves):
        self.inputs = tokenized_inputs
        self.outputs = output_moves
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        inp = encode(self.inputs[idx])
        out = stoi.get(self.outputs[idx], stoi["<unk>"])
        mask = legal_moves_from_sequence(self.inputs[idx])
        
        return inp, torch.tensor(out), mask

def collate_fn(batch):
    """Collate function for DataLoader"""
    inputs, outputs, masks = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in inputs])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=stoi["<pad>"])
    masks = torch.stack(masks)
    outputs = torch.stack(outputs)
    
    return padded_inputs, lengths, outputs, masks
