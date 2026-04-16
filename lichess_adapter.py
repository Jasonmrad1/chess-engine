"""
Lichess API adapter and move/board helper functions.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterator, List, Optional, Tuple

import berserk
import chess
from dotenv import load_dotenv
load_dotenv()
SUPPORTED_VARIANTS = {"standard", "fromPosition"}


def split_uci_moves(moves_blob: str) -> List[str]:
    """Split a Lichess move string into UCI tokens."""
    if not moves_blob:
        return []
    return [token for token in moves_blob.strip().split(" ") if token]


def lichess_to_board_move(board: chess.Board, uci_move: str) -> chess.Move:
    """Convert a Lichess UCI move into a legal python-chess move."""
    try:
        move = chess.Move.from_uci(uci_move)
    except ValueError as exc:
        raise ValueError(f"Invalid UCI move from Lichess: {uci_move}") from exc

    if move not in board.legal_moves:
        raise ValueError(f"Illegal move for current board: {uci_move}")

    return move


def apply_move(board: chess.Board, uci_move: str) -> chess.Move:
    """Validate and apply one UCI move to a board."""
    move = lichess_to_board_move(board, uci_move)
    board.push(move)
    return move


def is_our_turn(board: chess.Board, our_color: Optional[chess.Color]) -> bool:
    """Return True when it is our side to move and game is not over."""
    if our_color is None:
        return False
    return (not board.is_game_over()) and board.turn == our_color


def create_initial_board(initial_fen: str, chess960: bool = False) -> chess.Board:
    """Create initial board from Lichess initialFen semantics."""
    if not initial_fen or initial_fen == "startpos":
        return chess.Board(chess960=chess960)
    return chess.Board(initial_fen, chess960=chess960)


def build_board_from_moves(initial_fen: str, chess960: bool, uci_moves: List[str]) -> chess.Board:
    """Build a board from initial position plus full move list."""
    board = create_initial_board(initial_fen=initial_fen, chess960=chess960)
    for uci_move in uci_moves:
        apply_move(board, uci_move)
    return board


def infer_our_color(game_full: Dict, our_user_id: str) -> chess.Color:
    """Infer our color from gameFull payload and account id."""
    user_id = (our_user_id or "").lower()
    white_id = str(game_full.get("white", {}).get("id", "")).lower()
    black_id = str(game_full.get("black", {}).get("id", "")).lower()

    if white_id == user_id:
        return chess.WHITE
    if black_id == user_id:
        return chess.BLACK

    raise ValueError("Could not infer our color from game payload")


def opponent_name(game_full: Dict, our_color: chess.Color) -> str:
    """Return opponent display id from gameFull payload."""
    if our_color == chess.WHITE:
        return str(game_full.get("black", {}).get("id", "unknown"))
    return str(game_full.get("white", {}).get("id", "unknown"))


def challenge_supported(challenge: Dict) -> Tuple[bool, Optional[str]]:
    """Return whether a challenge should be accepted."""
    variant_key = str(challenge.get("variant", {}).get("key", "standard"))
    if variant_key not in SUPPORTED_VARIANTS:
        return False, "variant"
    return True, None


class LichessAdapter:
    """Thin berserk wrapper to centralize API usage and auth."""

    def __init__(self, token: str, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("lichess")
        self.token = token
        self.session: Optional[berserk.TokenSession] = None
        self.client: Optional[berserk.Client] = None
        self._rebuild_client()
        self.user_id = ""

    def _rebuild_client(self):
        self.session = berserk.TokenSession(self.token)
        self.client = berserk.Client(session=self.session)

    def reset_connection(self):
        """Recreate the underlying HTTP session/client after transport failures."""
        self._rebuild_client()
        self.logger.warning("Reset Lichess API session after network error.")

    @classmethod
    def from_env(cls, token_env: str = "LICHESS_TOKEN", logger: Optional[logging.Logger] = None) -> "LichessAdapter":
        token = os.environ.get(token_env, "").strip()
        if not token:
            raise RuntimeError(f"Missing required environment variable: {token_env}")
        return cls(token=token, logger=logger)

    def ensure_bot_account(self) -> str:
        """Upgrade account to bot if needed, then return our account id."""
        try:
            self.client.account.upgrade_to_bot()
            self.logger.info("Account upgraded to bot successfully.")
        except Exception as exc:
            msg = str(exc).lower()
            if "already" in msg and "bot" in msg:
                self.logger.info("Account is already a bot account.")
            else:
                self.logger.warning("Bot upgrade call returned non-fatal error: %s", exc)

        profile = self.client.account.get()
        self.user_id = str(profile.get("id", "")).lower()
        if not self.user_id:
            raise RuntimeError("Could not read account id from Lichess profile.")

        self.logger.info("Authenticated as bot account: %s", self.user_id)
        return self.user_id

    def stream_incoming_events(self) -> Iterator[Dict]:
        return self.client.bots.stream_incoming_events()

    def stream_game_state(self, game_id: str) -> Iterator[Dict]:
        return self.client.bots.stream_game_state(game_id)

    def accept_challenge(self, challenge_id: str):
        self.client.bots.accept_challenge(challenge_id)

    def decline_challenge(self, challenge_id: str, reason: str = "variant"):
        self.client.bots.decline_challenge(challenge_id, reason=reason)

    def make_move(self, game_id: str, move_uci: str):
        self.client.bots.make_move(game_id, move_uci)

    def resign_game(self, game_id: str):
        self.client.bots.resign_game(game_id)
