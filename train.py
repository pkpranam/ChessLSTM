import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import yaml

from data_utils import load_vocab, ChessDataset, collate_fn, stoi
from model import LSTMModel

def train():
    # load config from yaml
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['system']['device'] if torch.cuda.is_available() else 'cpu')
    
    # load data
    with open(cfg['data']['train_inputs'], "r") as f:
        train_input_lines = [line.strip() for line in f]
    with open(cfg['data']['val_inputs'], "r") as f:
        val_input_lines = [line.strip() for line in f]
    with open(cfg['data']['train_outputs'], "r") as f:
        train_output_lines = [line.strip() for line in f]
    with open(cfg['data']['val_outputs'], "r") as f:
        val_output_lines = [line.strip() for line in f]
    with open(cfg['data']['dataset'], "r") as f:
        dataset = [line.strip() for line in f]
    
    print(f"Train Inputs: {len(train_input_lines)}")
    print(f"Train Outputs: {len(train_output_lines)}")
    print(f"Val Inputs:   {len(val_input_lines)}")
    print(f"Val Outputs:  {len(val_output_lines)}")
    
    # build vocab
    vocab_size = load_vocab(dataset)
    print(f"Vocab size: {vocab_size}")
    
    # tokenize
    tokenized_train_inputs = [line.split(", ") for line in train_input_lines]
    tokenized_val_inputs = [line.split(", ") for line in val_input_lines]
    
    # filter by max length
    max_len = cfg['data']['max_len']
    tokenized_train_inputs_short = [tokenized_train_inputs[i] for i in range(len(tokenized_train_inputs)) if len(tokenized_train_inputs[i]) <= max_len]
    tokenized_val_inputs_short = [tokenized_val_inputs[i] for i in range(len(tokenized_val_inputs)) if len(tokenized_val_inputs[i]) <= max_len]
    train_output_lines_short = [train_output_lines[i] for i in range(len(train_output_lines)) if len(tokenized_train_inputs[i]) <= max_len]
    val_output_lines_short = [val_output_lines[i] for i in range(len(val_output_lines)) if len(tokenized_val_inputs[i]) <= max_len]
    
    # create datasets
    chess_train_dataset = ChessDataset(tokenized_train_inputs_short, train_output_lines_short)
    chess_val_dataset = ChessDataset(tokenized_val_inputs_short, val_output_lines_short)
    train_loader = DataLoader(chess_train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(chess_val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # init model
    model = LSTMModel(
        vocab_size=vocab_size,
        embed_dim=cfg['model']['embed_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        num_layers=cfg['model']['num_layers'],
        dropout=cfg['model']['dropout'],
        embed_dropout=cfg['model']['embed_dropout'],
        pad_idx=stoi["<pad>"]
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['max_lr'], weight_decay=cfg['training']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    
    # training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stopping_patience = 100
    early_stopping_counter = 0
    
    print(f"\nStarting training on {device}")
    print(f"Embed: {cfg['model']['embed_dim']}, Hidden: {cfg['model']['hidden_dim']}, Layers: {cfg['model']['num_layers']}\n")
    
    for epoch in range(cfg['training']['epochs']):
        start_time = time.time()
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, lengths, y, masks in train_loader:
            x, lengths, y, masks = x.to(device), lengths.to(device), y.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            with torch.autocast('cuda'):
                logits = model(x, lengths)
                masked_logits = logits + masks
                loss = criterion(masked_logits, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            preds = masked_logits.argmax(dim=-1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # validation
        model.eval()
        all_logits = []
        all_labels = []
        val_sample_count = 0
        val_loss = 0
        
        with torch.no_grad():
            for x, lengths, y, masks in val_loader:
                x = x.to(device)
                lengths = lengths.to(device)
                y = y.to(device)
                masks = masks.to(device)
                
                logits = model(x, lengths)
                masked_logits = logits + masks
                all_logits.append(masked_logits.cpu())
                all_labels.append(y.cpu())
                
                val_loss += criterion(masked_logits, y).item() * y.size(0)
                val_sample_count += y.size(0)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        val_acc = (all_logits.argmax(dim=-1) == all_labels).float().mean().item()
        val_loss = val_loss / val_sample_count
        
        scheduler.step(val_acc)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_train_loss = train_loss
            best_train_acc = train_acc
            early_stopping_counter = 0
            torch.save(model, cfg['system']['save_path'])
            print(f"  -> New Best Model! (Acc: {val_acc:.4f})")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break
    
    print("\n" + "="*50)
    print(f"Final Best Train Loss: {best_train_loss:.4f}")
    print(f"Final Best Train Acc: {best_train_acc:.4f}")
    print(f"Final Best Val Loss: {best_val_loss:.4f}")
    print(f"Final Best Val Acc: {best_val_acc:.4f}")
    print("="*50)

if __name__ == "__main__":
    train()
