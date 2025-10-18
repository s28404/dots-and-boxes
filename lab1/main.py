"""
Main runner for Dots and Boxes

Zasady gry (link): https://en.wikipedia.org/wiki/Dots_and_Boxes

Autorzy: Kajetan Frąckowiak

Instrukcja przygotowania środowiska: zobacz README.md w repozytorium.
"""

from dots_and_boxes import DotsAndBoxes
from agent import find_best_move


def print_board(game):
    """Simple function to visualize the current board state."""
    print("=" * (game.cols * 8 + 1))
    for r in range(game.rows):
        row_str = ""
        for c in range(game.cols):
            row_str += "•"
            line = game.h_lines[r][c]
            row_str += "═══════" if line == 1 else "───────" if line == 2 else "       "
        print(row_str + "•")

        row_str = ""
        for c in range(game.cols + 1):
            line = game.v_lines[r][c]
            row_str += "║" if line == 1 else "│" if line == 2 else " "
            if c < game.cols:
                box = game.boxes[r][c]
                row_str += (
                    "  [1]  " if box == 1 else "  [2]  " if box == 2 else "       "
                )
        print(row_str)

    row_str = ""
    for c in range(game.cols):
        row_str += "•"
        line = game.h_lines[game.rows][c]
        row_str += "═══════" if line == 1 else "───────" if line == 2 else "       "
    print(row_str + "•")
    print("=" * (game.cols * 8 + 1))
    print(f"Score - AI [1]: {game.score[1]} | Human [2]: {game.score[2]}")
    print("Legend: ═/║ = AI (Player 1) | ─/│ = Human (Player 2)\n")


def main():
    """The main function to run the interactive Dots and Boxes game."""
    BOARD_ROWS, BOARD_COLS, AI_DEPTH = 7, 5, 3
    game = DotsAndBoxes(BOARD_ROWS, BOARD_COLS)
    print(f"--- Starting Dots and Boxes ({BOARD_ROWS}x{BOARD_COLS} Boxes) ---")

    while not game.is_terminal():
        print_board(game)
        player = game.current_player

        if player == 1:
            print(f"AI (Player 1)'s turn (Searching depth {AI_DEPTH})...")
            best_move = find_best_move(game, AI_DEPTH)
            if best_move:
                line_type, r, c = best_move
                print(f"AI plays: {line_type} line at ({r}, {c})")
                if game.make_move(line_type, r, c, player):
                    print("AI completed a box and gets another turn!")
            else:
                print("Game error: No valid moves, but not terminal.")
                break
        else:
            print("Human (Player 2)'s turn.")
            moves = game.get_valid_moves()
            if not moves:
                print("No moves left for the human player. Passing turn...")
                game.current_player = 1
                continue

            while True:
                print("Enter move (e.g., h 0 0 for horizontal line at row 0, col 0):")
                try:
                    move_input = input("> ").split()
                    line_type, r, c = (
                        move_input[0].lower(),
                        int(move_input[1]),
                        int(move_input[2]),
                    )
                    if (line_type, r, c) in moves:
                        if game.make_move(line_type, r, c, player):
                            print("🎉 You completed a box and get another turn!")
                        break
                    else:
                        print("Invalid or already taken move. Try again.")
                except Exception:
                    print("Invalid input format. Use: [h/v] [row] [col]")

    print_board(game)
    print("--- GAME OVER ---")
    if game.score[1] > game.score[2]:
        print(f"AI (Player 1) WINS with {game.score[1]} to {game.score[2]}!")
    elif game.score[2] > game.score[1]:
        print(f"Human (Player 2) WINS with {game.score[2]} to {game.score[1]}!")
    else:
        print("It's a DRAW!")


if __name__ == "__main__":
    main()
