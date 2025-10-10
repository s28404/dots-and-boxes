"""
Dots and Boxes - core game implementation

Zasady gry (link): https://en.wikipedia.org/wiki/Dots_and_Boxes

Autorzy: (dodaj swoje imię/alias tutaj)

Instrukcja przygotowania środowiska: zobacz README.md w repozytorium.
"""


class DotsAndBoxes:
    """
    A class to represent the Dots and Boxes game board and logic.
    The board is a grid of R rows x C columns of boxes.
    """

    def __init__(self, rows=2, cols=2):
        self.rows = rows  # Number of rows of boxes
        self.cols = cols  # Number of columns of boxes

        # 2D array for horizontal lines: (rows+1) x cols
        # Value is 0 (unclaimed), 1 (Player 1/MAX), or 2 (Player 2/MIN)
        self.h_lines = [[0] * cols for _ in range(rows + 1)]

        # 2D array for vertical lines: rows x (cols+1)
        self.v_lines = [[0] * (cols + 1) for _ in range(rows)]

        # 2D array to track claimed boxes: rows x cols
        self.boxes = [[0] * cols for _ in range(rows)]

        self.current_player = 1  # 1 (MAX/AI) or 2 (MIN/Human)
        # Scores for both players, end if self.scores sum to rows*cols
        self.score = {1: 0, 2: 0}

    def get_valid_moves(self):
        """Returns a list of all legal moves (un-drawn lines)."""
        moves = []

        # Check horizontal lines
        for r in range(self.rows + 1):
            for c in range(self.cols):
                # If h_lines == 0, it's unclaimed
                if self.h_lines[r][c] == 0:
                    moves.append(("h", r, c))

        # Check vertical lines
        for r in range(self.rows):
            for c in range(self.cols + 1):
                if self.v_lines[r][c] == 0:
                    moves.append(("v", r, c))

        return moves

    def make_move(self, line_type, r, c, player):
        """
        Applies a move to the board.
        Returns True if a box was completed (player gets another turn), False otherwise.
        """
        box_completed = False

        if line_type == "h":
            self.h_lines[r][c] = player

            # Check box BELOW the line (if r is not the bottom border)
            if r < self.rows and self._is_box_complete(r, c):
                self.boxes[r][c] = player
                self.score[player] += 1
                box_completed = True

            # Check box ABOVE the line (if r is not the top border)
            if r > 0 and self._is_box_complete(r - 1, c):
                self.boxes[r - 1][c] = player
                self.score[player] += 1
                box_completed = True

        elif line_type == "v":
            self.v_lines[r][c] = player

            # Check box RIGHT of the line (if c is not the right border)
            if c < self.cols and self._is_box_complete(r, c):
                self.boxes[r][c] = player
                self.score[player] += 1
                box_completed = True

            # Check box LEFT of the line (if c is not the left border)
            if c > 0 and self._is_box_complete(r, c - 1):
                self.boxes[r][c - 1] = player
                self.score[player] += 1
                box_completed = True

        # Only switch player if NO box was completed
        if not box_completed:
            self.current_player = 3 - self.current_player  # Switches 1 to 2, and 2 to 1

        return box_completed

    def _is_box_complete(self, r, c):
        """Helper to check if a box at (r, c) is complete."""
        # Check top (H_line[r][c]), bottom (H_line[r+1][c]), left (V_line[r][c]), right (V_line[r][c+1])
        return (
            self.h_lines[r][c] != 0
            and self.h_lines[r + 1][c] != 0
            and self.v_lines[r][c] != 0
            and self.v_lines[r][c + 1] != 0
        )

    def is_terminal(self):
        """Returns True if the game is over (all boxes claimed)."""
        return (self.score[1] + self.score[2]) == (self.rows * self.cols)

    def get_utility(self, player):
        """Returns the utility (score difference) for the given player."""
        opponent = 3 - player
        return self.score[player] - self.score[opponent]
