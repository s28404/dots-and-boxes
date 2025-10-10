from easyAI import TwoPlayerGame, Negamax
import copy


class DotsAndBoxesAI(TwoPlayerGame):
    """Wrapper to make DotsAndBoxes compatible with easyAI."""

    def __init__(self, game_state):
        """Initialize with a DotsAndBoxes game state."""
        self.game_state = game_state
        self.players = [1, 2]
        self.current_player = game_state.current_player
        self._move_history = []

    def possible_moves(self):
        """Return list of possible moves."""
        return [f"{t},{r},{c}" for t, r, c in self.game_state.get_valid_moves()]

    def make_move(self, move):
        """Make a move on the board."""
        t, r, c = move.split(",")
        r, c = int(r), int(c)

        # Save state before move for unmake_move
        # row[:] creates a shallow copy of each row
        old_h_lines = [row[:] for row in self.game_state.h_lines]
        old_v_lines = [row[:] for row in self.game_state.v_lines]
        old_boxes = [row[:] for row in self.game_state.boxes]
        old_score = self.game_state.score.copy()
        old_player = self.game_state.current_player

        self._move_history.append(
            (move, old_h_lines, old_v_lines, old_boxes, old_score, old_player)
        )

        self.game_state.make_move(t, r, c, self.current_player)
        self.current_player = self.game_state.current_player

    def unmake_move(self, move):
        """Undo a move by restoring previous state."""
        if not self._move_history:
            return

        saved_move, old_h_lines, old_v_lines, old_boxes, old_score, old_player = (
            self._move_history.pop()
        )

        self.game_state.h_lines = old_h_lines
        self.game_state.v_lines = old_v_lines
        self.game_state.boxes = old_boxes
        self.game_state.score = old_score
        self.game_state.current_player = old_player
        self.current_player = old_player

    def lose(self):
        """Check if current player has lost."""
        return (
            self.game_state.is_terminal()
            and self.game_state.score[self.current_player]
            < self.game_state.score[3 - self.current_player]
        )

    def is_over(self):
        """Check if game is over."""
        return self.game_state.is_terminal()

    def scoring(self):
        """Return score from current player's perspective."""
        return self.game_state.get_utility(self.current_player)


def find_best_move(game_state, depth):
    """
    Find the best move using easyAI's Negamax (alpha-beta pruning).

    Args:
        game_state: DotsAndBoxes instance
        depth: Search depth

    Returns:
        Best move as tuple (line_type, r, c)
    """
    wrapper = DotsAndBoxesAI(copy.deepcopy(game_state))
    move_str = Negamax(depth)(wrapper)
    if move_str:
        t, r, c = move_str.split(",")
        return (t, int(r), int(c))
    return game_state.get_valid_moves()[0] if game_state.get_valid_moves() else None
