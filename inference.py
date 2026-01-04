import torch
import yaml
from data_utils import encode, legal_moves_from_sequence, stoi, itos, load_vocab

def predict_next(model, moves, device, k=5):
    """Predict next chess move given sequence of moves"""
    model.eval()
    with torch.no_grad():
        seq = encode(moves).unsqueeze(0).to(device)
        lengths = torch.tensor([len(moves)]).to(device)
        logits = model(seq, lengths)
        
        mask = legal_moves_from_sequence(moves).to(logits.device)
        masked_logits = logits + mask
        
        topk_logits, topk_indices = torch.topk(masked_logits, k, dim=-1)
        probs = torch.softmax(topk_logits, dim=-1)
        pred_index = torch.multinomial(probs, num_samples=1).item()
        pred_id = topk_indices[0, pred_index].item()
        
        return itos[pred_id]

def play_game(model, device, initial_moves=None, num_moves=20):
    """Generate a chess game starting from initial position"""
    if initial_moves is None:
        moves = ['e4']
    else:
        moves = initial_moves.copy()
    
    print("Starting moves:", ", ".join(moves))
    print("\nGenerating moves:")
    
    for i in range(num_moves):
        next_move = predict_next(model, moves, device)
        moves.append(next_move)
        print(f"{i+1}. {next_move}")
    
    return moves

if __name__ == "__main__":
    # load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['system']['device'] if torch.cuda.is_available() else 'cpu')
    
    # load vocab first
    with open(cfg['data']['dataset'], "r") as f:
        dataset = [line.strip() for line in f]
    load_vocab(dataset)
    
    # load model
    model = torch.load(cfg['system']['save_path'], weights_only=False)
    model.to(device)
    model.eval()
    
    # test prediction
    print("Test 1: Predict next move")
    move_string = 'd4, Nf6, c4'
    moves = move_string.split(', ')
    next_move = predict_next(model, moves, device)
    print(f"Input: {move_string}")
    print(f"Predicted next move: {next_move}\n")
    
    # generate a game
    print("Test 2: Generate game from e4")
    play_game(model, device, initial_moves=['e4'], num_moves=20)
