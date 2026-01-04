import chess.pgn
import random

def write_san_to_file(pgn_path, dataset_path, name='carlsen', min_elo=2500, new_file=False):
    """Extract games from PGN file and write to dataset"""
    if new_file:
        dataset_append = "w"
    else:
        dataset_append = "a"
    
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file, \
         open(dataset_path, dataset_append, encoding="utf-8") as dataset_file:
        
        game_counter = 0
        kept_counter = 0
        
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_counter += 1
            
            headers = game.headers
            white = headers.get("White", "")
            black = headers.get("Black", "")
            
            # check if target player is in game
            white_is_target = name in white.lower()
            black_is_target = name in black.lower()
            
            if white_is_target and black_is_target:
                continue
            if not (white_is_target or black_is_target):
                continue
            
            # get opponent elo
            if white_is_target:
                opponent_elo_str = headers.get("BlackElo", "")
            else:
                opponent_elo_str = headers.get("WhiteElo", "")
            
            try:
                opponent_elo = int(opponent_elo_str)
            except ValueError:
                continue
            
            if opponent_elo < min_elo:
                continue
            
            # convert moves to SAN
            board = game.board()
            moves_san = []
            for move in game.mainline_moves():
                san = board.san(move)
                moves_san.append(san)
                board.push(move)
            
            dataset_file.write(", ".join(moves_san) + "\n")
            kept_counter += 1
        
        print(f"Wrote {kept_counter} high ELO games out of {game_counter} total games for {name}")

def create_train_val(dataset_path, train_input_path, train_output_path, 
                     val_input_path, val_output_path, max_len=50, 
                     num_samples=10, seed=67, val_ratio=0.2):
    """Split dataset into train/val with input/output pairs"""
    random.seed(seed)
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        games = [line.strip() for line in f if line.strip()]
    
    random.shuffle(games)
    
    num_games = len(games)
    num_val_games = int(num_games * val_ratio)
    
    val_games = games[:num_val_games]
    train_games = games[num_val_games:]
    
    print(f"Total games: {num_games}")
    print(f"Train games: {len(train_games)}")
    print(f"Validation games: {len(val_games)}")
    
    def process_games(game_list, input_file, output_file):
        total_entries = 0
        
        for game_idx, game_line in enumerate(game_list, start=1):
            moves = [m.strip() for m in game_line.split(",") if m.strip()]
            num_moves = len(moves)
            
            if num_moves < 2:
                continue
            
            for _ in range(num_samples):
                k = random.randint(1, min(num_moves - 1, max_len))
                
                input_seq = ", ".join(moves[:k])
                next_move = moves[k]
                
                input_file.write(input_seq + "\n")
                output_file.write(next_move + "\n")
                
                total_entries += 1
            
            if game_idx % 1000 == 0:
                print(f"Processed {game_idx} games... entries so far: {total_entries}")
        
        return total_entries
    
    with open(train_input_path, "w", encoding="utf-8") as train_in, \
         open(train_output_path, "w", encoding="utf-8") as train_out, \
         open(val_input_path, "w", encoding="utf-8") as val_in, \
         open(val_output_path, "w", encoding="utf-8") as val_out:
        
        train_entries = process_games(train_games, train_in, train_out)
        val_entries = process_games(val_games, val_in, val_out)
    
    print("\nDone!")
    print(f"Train entries: {train_entries}")
    print(f"Val entries: {val_entries}")

if __name__ == "__main__":
    dataset_path = './data/dataset.txt'
    
    # extract games from PGN files
    players = [
        ('Carlsen.pgn', 'carlsen'),
        ('Kasparov.pgn', 'kasparov'),
        ('Caruana.pgn', 'caruana'),
        ('Aronian.pgn', 'aronian'),
        ('Mamedyarov.pgn', 'mamedyarov'),
        ('Anand.pgn', 'anand'),
        ('Ding.pgn', 'ding'),
        ('Kramnik.pgn', 'kramnik'),
        ('Topalov.pgn', 'topalov'),
        ('Firouzja.pgn', 'firouzja'),
        ('Nakamura.pgn', 'nakamura'),
        ('Grischuk.pgn', 'grischuk'),
        ('Gelfand.pgn', 'gelfand'),
        ('Giri.pgn', 'giri'),
        ('Gukesh.pgn', 'gukesh'),
        ('Harikrishna.pgn', 'harikrishna'),
        ('Ivanchuk.pgn', 'ivanchuk'),
        ('Karpov.pgn', 'karpov'),
        ('Nepomniachtchi.pgn', 'nepomniachtchi'),
        ('Praggnanandhaa.pgn', 'praggnanandhaa'),
        ('Radjabov.pgn', 'radjabov'),
        ('Rapport.pgn', 'rapport')
    ]
    
    for i, (pgn_file, player_name) in enumerate(players):
        write_san_to_file(
            pgn_path=f'./pgns/{pgn_file}',
            dataset_path=dataset_path,
            name=player_name,
            min_elo=2500,
            new_file=(i == 0)
        )
    
    # create train/val splits
    train_input_path = './data/train_inputs.txt'
    train_output_path = './data/train_outputs.txt'
    val_input_path = './data/val_inputs.txt'
    val_output_path = './data/val_outputs.txt'
    
    create_train_val(
        dataset_path=dataset_path,
        train_input_path=train_input_path,
        train_output_path=train_output_path,
        val_input_path=val_input_path,
        val_output_path=val_output_path
    )
