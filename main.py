"""
Main runner for Dots and Boxes

Zasady gry (link): https://en.wikipedia.org/wiki/Dots_and_Boxes

Autorzy: Kajetan Frąckowiak

Instrukcja przygotowania środowiska: zobacz README.md w repozytorium.
"""

from dots_and_boxes import DotsAndBoxes
from minimax_alpha_beta import find_best_move

# Assuming the DotsAndBoxes class and find_best_move are defined above.
# moves h 0 0 makes line from dot (0,0) to (0,1)
# moves v 0 0 makes line from dot (0,0) to (1,0)


def print_board(game):
    """Simple function to visualize the current board state."""
    print("=" * (game.cols * 8 + 1))
    for r in range(game.rows):
        # Print horizontal lines (dots and top/middle lines)
        row_str = ""
        for c in range(game.cols):
            row_str += "•"
            line_status = game.h_lines[r][c]
            if line_status == 1:
                row_str += "═══════"  # AI/Player 1 uses double lines
            elif line_status == 2:
                row_str += "───────"  # Human/Player 2 uses single lines
            else:
                row_str += "       "  # Unclaimed line
        row_str += "•"
        print(row_str)

        # Print vertical lines and boxes
        row_str = ""
        for c in range(game.cols + 1):
            line_status = game.v_lines[r][c]
            if line_status == 1:
                row_str += "║"  # AI/Player 1 uses double vertical
            elif line_status == 2:
                row_str += "│"  # Human/Player 2 uses single vertical
            else:
                row_str += " "  # Unclaimed line

            if c < game.cols:
                box_owner = game.boxes[r][c]
                if box_owner == 1:
                    row_str += "  [1]  "  # AI box
                elif box_owner == 2:
                    row_str += "  [2]  "  # Human box
                else:
                    row_str += "       "  # Unclaimed box
        print(row_str)

    # Print the final horizontal lines (bottom border)
    row_str = ""
    for c in range(game.cols):
        row_str += "•"
        line_status = game.h_lines[game.rows][c]
        if line_status == 1:
            row_str += "═══════"  # AI/Player 1
        elif line_status == 2:
            row_str += "───────"  # Human/Player 2
        else:
            row_str += "       "  # Unclaimed
    row_str += "•"
    print(row_str)
    print("=" * (game.cols * 8 + 1))
    print(f"Score - AI [1]: {game.score[1]} | Human [2]: {game.score[2]}")
    print("Legend: ═/║ = AI (Player 1) | ─/│ = Human (Player 2)")
    print("\n")


def main():
    """The main function to run the interactive Dots and Boxes game."""

    # 1. Initialization
    BOARD_ROWS = 7  # 7x5 box grid (8x6 dots)
    BOARD_COLS = 5
    AI_DEPTH = 3  # Search depth for the Alpha-Beta AI. Reduced for larger board.

    game = DotsAndBoxes(BOARD_ROWS, BOARD_COLS)
    print(f"--- Starting Dots and Boxes ({BOARD_ROWS}x{BOARD_COLS} Boxes) ---")

    # 2. Game Loop
    while not game.is_terminal():
        print_board(game)

        player = game.current_player

        # Determine if the current player is the AI (Player 1) or Human (Player 2)
        if player == 1:
            # === AI (MAX) Turn ===
            print(f"AI (Player 1)'s turn (Searching depth {AI_DEPTH})...")

            # Find the best move using the Alpha-Beta algorithm
            best_move = find_best_move(game, AI_DEPTH)

            if best_move:
                line_type, r, c = best_move
                print(f"AI plays: {line_type} line at ({r}, {c})")

                # Apply the move and check if AI gets another turn
                box_completed = game.make_move(line_type, r, c, player)

                if box_completed:
                    print("AI completed a box and gets another turn!")
            else:
                print("Game error: No valid moves, but not terminal.")
                break  # Should not happen if is_terminal is correct

        else:
            # === Human (MIN) Turn ===
            print("Human (Player 2)'s turn.")
            moves = game.get_valid_moves()

            if not moves:
                print("No moves left for the human player. Passing turn...")
                game.current_player = (
                    1  # Switch back to AI if somehow no moves are left
                )
                continue

            # Prompt for human input
            while True:
                print("Enter move (e.g., h 0 0 for horizontal line at row 0, col 0):")
                try:
                    move_input = input("> ").split()
                    line_type, r_str, c_str = move_input
                    r, c = int(r_str), int(c_str)
                    move = (line_type.lower(), r, c)

                    if move in moves:
                        # Apply the move and check if Human gets another turn
                        box_completed = game.make_move(line_type.lower(), r, c, player)
                        if box_completed:
                            print("🎉 You completed a box and get another turn!")
                        break  # Exit the input loop
                    else:
                        print("Invalid or already taken move. Try again.")
                except Exception:
                    print("Invalid input format. Use: [h/v] [row] [col]")

    # 3. Game End
    print_board(game)
    print("--- GAME OVER ---")
    if game.score[1] > game.score[2]:
        print(f"AI (Player 1) WINS with {game.score[1]} to {game.score[2]}!")
    elif game.score[2] > game.score[1]:
        print(f"Human (Player 2) WINS with {game.score[2]} to {game.score[1]}!")
    else:
        print("It's a DRAW!")


# Entry point
if __name__ == "__main__":
    main()
