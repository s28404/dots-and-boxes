# --- Alpha-Beta Pruning Implementation ---

import copy # Used to create a new game state for recursion

# Use large numbers for initial alpha/beta
INF = float('inf')

def minimax_alpha_beta(game_state, depth, alpha, beta, maximizing_player):
    """
    Minimax algorithm with Alpha-Beta Pruning.
    """
    
    # 1. Base Case: Check for end of game or depth limit
    if depth == 0 or game_state.is_terminal():
        # The utility must be calculated from the perspective of the initial MAX player (Player 1)
        # We assume Player 1 is the maximizing player (AI).
        return game_state.get_utility(1)

    # 2. Recursive Step
    if maximizing_player:
        max_eval = -INF
        # Moves: [('h' or 'v', row, col), ...]
        moves = game_state.get_valid_moves()
        
        for move in moves:
            # Create a deep copy of the game state to simulate the move
            new_state = copy.deepcopy(game_state)
            
            line_type, r, c = move
            box_completed = new_state.make_move(line_type, r, c, new_state.current_player)
            
            # The player does NOT switch if a box was completed
            next_maximizing_player = maximizing_player if box_completed else False 

            # Recursive call
            # Depth - 1 because we move down one level in the game tree
            evaluation = minimax_alpha_beta(new_state, depth - 1, alpha, beta, next_maximizing_player)
            
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            
            # Alpha-Beta Pruning check
            if beta <= alpha:
                break
        return max_eval
        
    else: # Minimizing player (Human)
        min_eval = INF
        moves = game_state.get_valid_moves()

        for move in moves:
            # Create a deep copy of the game state to simulate the move
            new_state = copy.deepcopy(game_state)
            
            line_type, r, c = move
            box_completed = new_state.make_move(line_type, r, c, new_state.current_player)
            
            # The player does NOT switch if a box was completed
            next_maximizing_player = maximizing_player if box_completed else True

            # Recursive call
            evaluation = minimax_alpha_beta(new_state, depth - 1, alpha, beta, next_maximizing_player)
            
            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)
            
            # Alpha-Beta Pruning check
            if beta <= alpha:
                break
        return min_eval

def find_best_move(game_state, depth):
    """Iterates through all possible moves and finds the one with the best Minimax value."""
    best_move = None
    best_eval = -INF
    
    for move in game_state.get_valid_moves():
        temp_state = copy.deepcopy(game_state)
        line_type, r, c = move

        # Make the move in the temporary state
        box_completed = temp_state.make_move(line_type, r, c, temp_state.current_player)
        
        # Determine the next player's role in the recursive call
        next_maximizing_player = True if temp_state.current_player == 1 else False
        
        # Use Alpha-Beta to evaluate the resulting state
        eval = minimax_alpha_beta(temp_state, depth - 1, -INF, INF, next_maximizing_player)

        if eval > best_eval:
            best_eval = eval
            best_move = move
            
    return best_move