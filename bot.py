"""
Lichess bot runner using berserk + engine.find_best_move.

Requirements:
- Set LICHESS_TOKEN in your environment
- pip install berserk python-chess
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chess

import engine
from lichess_adapter import (
    LichessAdapter,
    apply_move,
    build_board_from_moves,
    challenge_supported,
    infer_our_color,
    is_our_turn,
    opponent_name,
    split_uci_moves,
)


LOGGER = logging.getLogger("lichess-bot")


PIECE_VALUE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 950,
    chess.KING: 20000,
}


def is_simple_position(board: chess.Board) -> bool:
    """
    Lightweight heuristic to detect simple positions without engine.
    Returns True if position is likely solvable quickly.
    """
    # Very few legal moves = forced line
    legal_moves = list(board.legal_moves)
    if len(legal_moves) <= 2 and chess.popcount(board.occupied) <= 6:
        return True

    # In-check positions are usually tactical: do not classify as simple.
    if board.is_check():
        return False

    # Very few forcing moves (captures or checks)
    forcing_moves = [
        m for m in legal_moves
        if board.is_capture(m) or board.gives_check(m)
    ]
    if len(forcing_moves) <= 1 and len(legal_moves) <= 12:
        return True

    return False


@dataclass
class GameContext:
    game_id: str
    initial_fen: str = "startpos"
    chess960: bool = False
    board: chess.Board = field(default_factory=chess.Board)
    our_color: Optional[chess.Color] = None
    opponent_id: str = "unknown"
    seen_moves: List[str] = field(default_factory=list)
    finished: bool = False

    # ✅ ADDED CLOCK FIELDS
    wtime: int = 0
    btime: int = 0
    winc: int = 0
    binc: int = 0
    needs_resync: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


class LichessBotRunner:
    def __init__(
        self,
        adapter: LichessAdapter,
        *,
        search_verbose: bool = False,
        use_fast_eval_in_complex: bool = True,
    ):
        self.adapter = adapter
        self.stop_event = threading.Event()
        self.search_verbose = search_verbose
        self.use_fast_eval_in_complex = use_fast_eval_in_complex

        self.games: Dict[str, GameContext] = {}
        self.game_threads: Dict[str, threading.Thread] = {}
        self.games_lock = threading.Lock()

        # engine.find_best_move uses shared state in engine.py; lock for thread safety.
        self.engine_lock = threading.Lock()

    @staticmethod
    def _clock_seconds(raw_value: object) -> float:
        if isinstance(raw_value, (int, float)):
            return max(0.0, float(raw_value) / 1000.0)
        total_seconds = getattr(raw_value, "total_seconds", None)
        if callable(total_seconds):
            return max(0.0, float(total_seconds()))
        return 0.0

    @staticmethod
    def _hard_think_cap(remaining: float, increment: float) -> float:
        inc = max(0.0, increment)
        if remaining <= 5:
            return max(0.10, min(0.60, 0.12 + 0.10 * inc))
        if remaining <= 10:
            return max(0.20, min(1.00, 0.25 + 0.20 * inc))
        if remaining <= 20:
            return max(0.40, min(2.20, 0.65 + 0.35 * inc))
        if remaining <= 40:
            return max(0.70, min(3.40, 1.30 + 0.45 * inc))
        if remaining <= 90:
            return max(1.20, min(5.50, 2.00 + 0.55 * inc))
        if remaining <= 180:
            return max(1.80, min(8.50, 2.40 + 0.80 * inc + 0.015 * remaining))
        if remaining <= 300:
            return max(2.50, min(12.50, 3.40 + 0.90 * inc + 0.020 * remaining))
        if remaining <= 600:
            return max(4.00, min(18.00, 5.00 + 1.00 * inc + 0.018 * remaining))
        return min(26.0, 7.00 + 1.10 * inc + 0.015 * remaining)

    @staticmethod
    def _clock_profile(remaining: float, increment: float) -> str:
        # Effective clock tuned so 5+3 is treated as blitz-like for safer time usage.
        effective = remaining + 25.0 * max(0.0, increment)
        if effective <= 120:
            return "bullet"
        if effective <= 420:
            return "blitz"
        if effective <= 1200:
            return "rapid"
        return "classical"

    @staticmethod
    def _reserve_floor(profile: str, increment: float, seen_moves: int) -> float:
        inc = max(0.0, increment)
        if profile == "bullet":
            reserve = 1.4 + 2.0 * inc
        elif profile == "blitz":
            reserve = 3.2 + 3.2 * inc
        elif profile == "rapid":
            reserve = 5.0 + 3.8 * inc
        else:
            reserve = 7.0 + 4.2 * inc
        # Keep a little extra early-game safety to absorb network jitter/retries.
        if seen_moves < 14:
            reserve += 1.0
        return max(0.8, reserve)

    @staticmethod
    def _profile_clock_cap(profile: str, remaining: float, increment: float) -> float:
        inc = max(0.0, increment)
        if profile == "bullet":
            return max(0.05, 0.05 + 0.006 * remaining + 0.18 * inc)
        if profile == "blitz":
            return max(0.06, 0.18 + 0.010 * remaining + 0.30 * inc)
        if profile == "rapid":
            return max(0.08, 0.24 + 0.011 * remaining + 0.36 * inc)
        return max(0.10, 0.30 + 0.013 * remaining + 0.45 * inc)

    @staticmethod
    def _best_capture_swing(board: chess.Board) -> int:
        best = -10**9
        any_capture = False
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
            any_capture = True

            attacker = board.piece_at(move.from_square)
            if attacker is None:
                continue

            if board.is_en_passant(move):
                captured_value = PIECE_VALUE_CP[chess.PAWN]
            else:
                captured_piece = board.piece_at(move.to_square)
                if captured_piece is None:
                    continue
                captured_value = PIECE_VALUE_CP.get(captured_piece.piece_type, 0)

            attacker_value = PIECE_VALUE_CP.get(attacker.piece_type, 0)
            swing = captured_value - attacker_value
            if move.promotion:
                swing += PIECE_VALUE_CP.get(move.promotion, 0) - PIECE_VALUE_CP[chess.PAWN]
            if board.gives_check(move):
                swing += 25

            if swing > best:
                best = swing

        if not any_capture:
            return -10**9
        return best

    @staticmethod
    def _dynamic_depth_cap(
        remaining: float,
        increment: float,
        simple: bool,
        piece_count: int,
        legal_count: int,
    ) -> int:
        depth = 24 if simple else 30

        if remaining >= 600:
            depth += 10
        elif remaining >= 300:
            depth += 8
        elif remaining >= 180:
            depth += 6
        elif remaining >= 90:
            depth += 4
        elif remaining >= 45:
            depth += 2

        if increment >= 8:
            depth += 2
        elif increment >= 3:
            depth += 1

        if piece_count <= 12:
            depth += 3
        elif piece_count <= 18:
            depth += 2
        elif piece_count >= 28 and not simple:
            depth -= 1

        if legal_count >= 42 and not simple:
            depth += 1

        if remaining < 8:
            depth = min(depth, 16)
        elif remaining < 15:
            depth = min(depth, 18)
        elif remaining < 30:
            depth = min(depth, 22)
        elif remaining < 60:
            depth = min(depth, 26)

        return max(12, min(44, depth))

    @staticmethod
    def _should_use_fast_eval(simple: bool, piece_count: int, legal_count: int) -> bool:
        # Middlegame-like, high-branching positions benefit most from faster eval.
        return (not simple) and piece_count >= 18 and legal_count >= 24

    def _deterministic_fallback_move(self, board: chess.Board) -> Optional[chess.Move]:
        legal_moves = sorted(board.legal_moves, key=lambda m: m.uci())
        if not legal_moves:
            return None

        best_move = legal_moves[0]
        best_score = -engine.INF
        for move in legal_moves:
            board.push(move)
            try:
                if board.is_checkmate():
                    score = engine.INF
                elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                    score = 0
                else:
                    with self.engine_lock:
                        score = -engine.ENGINE._eval_stm(board)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def run(self):
        self.adapter.ensure_bot_account()

        retry_delay = 1
        while not self.stop_event.is_set():
            try:
                LOGGER.info("Streaming incoming Lichess events...")
                for event in self.adapter.stream_incoming_events():
                    self.handle_incoming_event(event)
                    if self.stop_event.is_set():
                        break

                if self.stop_event.is_set():
                    break

                LOGGER.warning("Incoming event stream closed. Reconnecting...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
            except Exception:
                LOGGER.exception("Incoming event stream failed. Reconnecting in %ss", retry_delay)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
            else:
                retry_delay = 1

    def stop(self):
        self.stop_event.set()

    def handle_incoming_event(self, event: Dict):
        event_type = event.get("type")

        if event_type == "challenge":
            self._handle_challenge(event.get("challenge", {}))
            return

        if event_type == "gameStart":
            game_data = event.get("game", {})
            game_id = str(game_data.get("gameId") or game_data.get("id") or "").strip()
            if not game_id:
                LOGGER.error("Received gameStart event without game id: %s", event)
                return
            self._start_game_worker(game_id)
            return

        if event_type == "gameFinish":
            game_data = event.get("game", {})
            game_id = str(game_data.get("gameId") or game_data.get("id") or "").strip()
            if game_id:
                with self.games_lock:
                    ctx = self.games.get(game_id)
                    if ctx is not None:
                        with ctx.lock:
                            ctx.finished = True
                LOGGER.info("Game finished: %s", game_id)
            return

        LOGGER.debug("Ignoring incoming event type=%s", event_type)

    def _handle_challenge(self, challenge: Dict):
        challenge_id = str(challenge.get("id", "")).strip()
        if not challenge_id:
            LOGGER.error("Received challenge event without id: %s", challenge)
            return

        allowed, _ = challenge_supported(challenge)
        challenger = str(challenge.get("challenger", {}).get("id", "unknown"))
        variant = str(challenge.get("variant", {}).get("key", "standard"))

        try:
            self.adapter.accept_challenge(challenge_id)
            if allowed:
                LOGGER.info("Accepted challenge %s from %s (variant=%s)", challenge_id, challenger, variant)
            else:
                LOGGER.warning(
                    "Accepted challenge %s from %s (unsupported variant=%s). "
                    "Game worker will resign safely.",
                    challenge_id,
                    challenger,
                    variant,
                )
        except Exception as e:
            # ✅ Gracefully ignore 404 errors (expired or already-accepted challenges)
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str:
                LOGGER.warning("Challenge %s already handled or expired (404), ignoring.", challenge_id)
            else:
                LOGGER.exception("Failed to process challenge %s", challenge_id)

    def _start_game_worker(self, game_id: str):
        with self.games_lock:
            existing = self.game_threads.get(game_id)
            if existing is not None and existing.is_alive():
                LOGGER.info("Game worker already running for %s", game_id)
                return

            ctx = GameContext(game_id=game_id)
            self.games[game_id] = ctx

            thread = threading.Thread(
                target=self._game_loop,
                args=(game_id,),
                name=f"game-{game_id[:8]}",
                daemon=True,
            )
            self.game_threads[game_id] = thread
            thread.start()

    def _game_loop(self, game_id: str):
        LOGGER.info("Game stream started: %s", game_id)
        retry_delay = 1

        while not self.stop_event.is_set():
            with self.games_lock:
                ctx = self.games.get(game_id)

            if ctx is None:
                break
            with ctx.lock:
                if ctx.finished:
                    break

            try:
                for event in self.adapter.stream_game_state(game_id):
                    self._handle_game_event(ctx, event)
                    with ctx.lock:
                        finished = ctx.finished
                    if finished or self.stop_event.is_set():
                        break

                with ctx.lock:
                    finished = ctx.finished
                if finished or self.stop_event.is_set():
                    break

                LOGGER.warning("Game stream ended unexpectedly for %s. Reconnecting...", game_id)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
            except Exception:
                with ctx.lock:
                    finished = ctx.finished
                if finished or self.stop_event.is_set():
                    break
                LOGGER.exception("Game stream failed for %s. Reconnecting in %ss", game_id, retry_delay)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
            else:
                retry_delay = 1

        with self.games_lock:
            self.games.pop(game_id, None)
            self.game_threads.pop(game_id, None)

        LOGGER.info("Game worker stopped: %s", game_id)

    def _handle_game_event(self, ctx: GameContext, event: Dict):
        event_type = event.get("type")

        if event_type == "gameFull":
            self._handle_game_full(ctx, event)
            return

        if event_type == "gameState":
            self._sync_state_and_maybe_move(ctx, event)
            return

        if event_type in {"chatLine", "opponentGone"}:
            return

        LOGGER.debug("Game %s: unhandled event type=%s", ctx.game_id, event_type)

    def _handle_game_full(self, ctx: GameContext, game_full: Dict):
        variant_key = str(game_full.get("variant", {}).get("key", "standard"))
        if variant_key not in {"standard", "fromPosition"}:
            LOGGER.error("Game %s uses unsupported variant=%s. Resigning.", ctx.game_id, variant_key)
            self._safe_resign(ctx)
            return

        with ctx.lock:
            ctx.chess960 = bool(game_full.get("chess960", False))
            ctx.initial_fen = str(game_full.get("initialFen", "startpos"))
            ctx.board = build_board_from_moves(ctx.initial_fen, ctx.chess960, [])
            ctx.seen_moves = []
            ctx.needs_resync = False

        try:
            our_color = infer_our_color(game_full, self.adapter.user_id)
            with ctx.lock:
                ctx.our_color = our_color
        except Exception:
            LOGGER.exception("Could not determine our color in game %s", ctx.game_id)
            self._safe_resign(ctx)
            return

        with ctx.lock:
            ctx.opponent_id = opponent_name(game_full, ctx.our_color)
        LOGGER.info(
            "Game start: %s | color=%s | opponent=%s",
            ctx.game_id,
            "white" if our_color == chess.WHITE else "black",
            ctx.opponent_id,
        )

        state = game_full.get("state", {})
        self._sync_state_and_maybe_move(ctx, state)

    def _sync_state_and_maybe_move(self, ctx: GameContext, state_payload: Dict):
        moves_blob = str(state_payload.get("moves", ""))
        incoming_moves = split_uci_moves(moves_blob)
        status = str(state_payload.get("status", "started"))

        should_play = False
        should_resign = False
        with ctx.lock:
            # Preserve previous clock values when payload omits any field.
            for field_name in ("wtime", "btime", "winc", "binc"):
                incoming = state_payload.get(field_name)
                if isinstance(incoming, (int, float)):
                    setattr(ctx, field_name, max(0, int(incoming)))

            try:
                if ctx.needs_resync:
                    ctx.board = build_board_from_moves(ctx.initial_fen, ctx.chess960, incoming_moves)
                    ctx.seen_moves = incoming_moves
                    ctx.needs_resync = False
                    LOGGER.warning("Game %s | forced resync completed.", ctx.game_id)
                elif len(incoming_moves) >= len(ctx.seen_moves) and incoming_moves[: len(ctx.seen_moves)] == ctx.seen_moves:
                    new_moves = incoming_moves[len(ctx.seen_moves) :]
                    for uci_move in new_moves:
                        mover = ctx.board.turn
                        apply_move(ctx.board, uci_move)
                        ctx.seen_moves.append(uci_move)

                        if ctx.our_color is not None and mover != ctx.our_color:
                            LOGGER.info("Game %s | opponent move: %s", ctx.game_id, uci_move)
                else:
                    ctx.board = build_board_from_moves(ctx.initial_fen, ctx.chess960, incoming_moves)
                    ctx.seen_moves = incoming_moves
                    LOGGER.info("Game %s | board resynced from move list.", ctx.game_id)
            except Exception:
                LOGGER.exception("Game %s | failed to apply incoming moves; forcing resync.", ctx.game_id)
                try:
                    ctx.board = build_board_from_moves(ctx.initial_fen, ctx.chess960, incoming_moves)
                    ctx.seen_moves = incoming_moves
                    ctx.needs_resync = False
                except Exception:
                    LOGGER.exception("Game %s | hard resync failed. Resigning for safety.", ctx.game_id)
                    should_resign = True

            if not should_resign:
                if status != "started":
                    ctx.finished = True
                    LOGGER.info("Game %s finished with status=%s", ctx.game_id, status)
                else:
                    should_play = is_our_turn(ctx.board, ctx.our_color)

        if should_resign:
            self._safe_resign(ctx)
            return

        if status != "started":
            return

        if should_play:
            self._play_engine_move(ctx)

    def _compute_time_for_move(self, ctx: GameContext) -> float:
        with ctx.lock:
            our_color = ctx.our_color
            wtime = ctx.wtime
            btime = ctx.btime
            winc = ctx.winc
            binc = ctx.binc
            seen_moves = len(ctx.seen_moves)
            board_snapshot = ctx.board.copy(stack=False)

        if our_color == chess.WHITE:
            remaining = self._clock_seconds(wtime)
            increment = self._clock_seconds(winc)
        else:
            remaining = self._clock_seconds(btime)
            increment = self._clock_seconds(binc)
        profile = self._clock_profile(remaining, increment)

        if remaining <= 0 and increment <= 0:
            return 0.05
        if remaining <= 0:
            return max(0.05, min(0.20, 0.05 + 0.5 * increment))

        piece_count = chess.popcount(board_snapshot.occupied)
        legal_count = board_snapshot.legal_moves.count()
        tactical_swing = self._best_capture_swing(board_snapshot)

        # Moves-left estimate by clock profile.
        if profile == "bullet":
            moves_left = max(24, min(90, 68 - seen_moves // 2))
        elif profile == "blitz":
            moves_left = max(22, min(80, 62 - seen_moves // 2))
        elif profile == "rapid":
            moves_left = max(18, min(66, 54 - seen_moves // 2))
        else:
            moves_left = max(14, min(56, 46 - seen_moves // 2))

        # Base time allocation
        time_for_move = remaining / moves_left
        if profile == "bullet":
            time_for_move += 0.22 * increment
        elif profile == "blitz":
            time_for_move += 0.30 * increment
        elif profile == "rapid":
            time_for_move += 0.40 * increment
        else:
            time_for_move += 0.50 * increment

        # ✅ Opening fast play
        if seen_moves < 10:
            time_for_move *= 0.6

        # Adaptive safety margin
        if profile == "bullet":
            safety_margin = 0.82
        elif profile == "blitz":
            safety_margin = 0.88
        elif profile == "rapid":
            safety_margin = 0.92
        else:
            safety_margin = 0.95
        if remaining <= 20:
            safety_margin -= 0.03
        if remaining <= 10:
            safety_margin -= 0.03
        safety_margin = max(0.72, safety_margin)
        time_for_move *= safety_margin

        # Scale tactical/complexity boosts down when low on time. 
        urgency = 1.0
        if remaining <= 5:
            urgency = 0.20
        elif remaining <= 10:
            urgency = 0.35
        elif remaining <= 20:
            urgency = 0.55
        elif remaining <= 40:
            urgency = 0.75

        # Position complexity adjustment
        if legal_count > 42:
            time_for_move *= 1.0 + 0.22 * urgency
        elif legal_count > 30:
            time_for_move *= 1.0 + 0.12 * urgency
        elif legal_count < 8 and piece_count > 6:
            time_for_move *= 0.92

        # Tactical boost
        if board_snapshot.is_check():
            time_for_move *= 1.0 + 0.30 * urgency

        if any(board_snapshot.is_capture(m) for m in board_snapshot.legal_moves):
            time_for_move *= 1.0 + 0.15 * urgency

        # If there's a clear material-gain capture available, avoid rushing this move.
        if tactical_swing >= 250:
            time_for_move *= 1.22 if remaining > 12 else 1.10
        elif tactical_swing >= 150:
            time_for_move *= 1.12 if remaining > 10 else 1.06

        # Endgame bonus
        if piece_count <= 12:
            time_for_move *= 1.0 + 0.08 * urgency

        # Panic mode
        if remaining <= 2.0:
            return max(0.04, min(0.10, 0.04 + 0.20 * increment))
        if remaining <= 4.0:
            return max(0.05, min(0.16, 0.05 + 0.35 * increment))
        if remaining <= 8.0:
            return max(0.06, min(0.30, 0.08 + 0.45 * increment))

        cap_from_clock = self._profile_clock_cap(profile, remaining, increment)

        # Don't spend so much that we crash below a safe reserve.
        reserve_floor = self._reserve_floor(profile, increment, seen_moves)
        if remaining > reserve_floor:
            if profile == "bullet":
                reserve_cap = (remaining - reserve_floor) * 0.10
            elif profile == "blitz":
                reserve_cap = (remaining - reserve_floor) * 0.14
            elif profile == "rapid":
                reserve_cap = (remaining - reserve_floor) * 0.17
            else:
                reserve_cap = (remaining - reserve_floor) * 0.20
        else:
            reserve_cap = max(0.04, 0.05 + 0.18 * increment)

        hard_cap = self._hard_think_cap(remaining, increment)
        min_budget = 0.04 if profile == "bullet" else 0.05
        return max(min_budget, min(time_for_move, cap_from_clock, reserve_cap, hard_cap))

    def _play_engine_move(self, ctx: GameContext):
        with ctx.lock:
            if ctx.our_color is None or ctx.finished:
                return

            if ctx.board.is_checkmate() and ctx.board.turn == ctx.our_color:
                should_resign = True
                position_for_engine = None
                our_color = ctx.our_color
            else:
                should_resign = False
                position_for_engine = ctx.board.copy(stack=True)
                our_color = ctx.our_color

        if should_resign:
            LOGGER.info("Game %s | checkmated position detected, resigning.", ctx.game_id)
            self._safe_resign(ctx)
            return

        # ✅ Detect simple positions using lightweight heuristic (no engine call)
        assert position_for_engine is not None
        simple = is_simple_position(position_for_engine)
        legal_count = position_for_engine.legal_moves.count()
        piece_count = chess.popcount(position_for_engine.occupied)
        tactical_swing = self._best_capture_swing(position_for_engine)

        # Compute base time allocation
        time_for_move = self._compute_time_for_move(ctx)

        with ctx.lock:
            if our_color == chess.WHITE:
                remaining = self._clock_seconds(ctx.wtime)
                increment = self._clock_seconds(ctx.winc)
            else:
                remaining = self._clock_seconds(ctx.btime)
                increment = self._clock_seconds(ctx.binc)
        seen_moves_now = len(position_for_engine.move_stack)
        profile = self._clock_profile(remaining, increment)

        max_depth = self._dynamic_depth_cap(
            remaining=remaining,
            increment=increment,
            simple=simple,
            piece_count=piece_count,
            legal_count=legal_count,
        )

        # Low-clock hard depth caps to avoid flags.
        if remaining <= 2:
            max_depth = min(max_depth, 9)
        elif remaining <= 5:
            max_depth = min(max_depth, 11)
        elif remaining <= 10:
            max_depth = min(max_depth, 13)
        elif remaining <= 20:
            max_depth = min(max_depth, 16)

        # Profile-aware safety caps (5+3 is blitz-like profile here).
        if profile == "blitz":
            if remaining <= 25:
                max_depth = min(max_depth, 14)
            elif remaining <= 60:
                max_depth = min(max_depth, 16)
            elif remaining <= 120:
                max_depth = min(max_depth, 18)
        elif profile == "rapid":
            if remaining <= 30:
                max_depth = min(max_depth, 16)
            elif remaining <= 90:
                max_depth = min(max_depth, 20)

        # Tactical guardrail: spend a bit more and search deeper on obvious material swings.
        if tactical_swing >= 250:
            max_depth = min(max_depth + (2 if remaining > 12 else 1), 24)
            time_for_move *= 1.20 if remaining > 12 else 1.08
        elif tactical_swing >= 150:
            max_depth = min(max_depth + 1, 22)
            time_for_move *= 1.10 if remaining > 10 else 1.05

        use_fast_eval = self.use_fast_eval_in_complex and self._should_use_fast_eval(
            simple=simple,
            piece_count=piece_count,
            legal_count=legal_count,
        )
        if tactical_swing >= 150 and remaining > 8:
            use_fast_eval = False
        elif remaining <= 12 or (profile == "blitz" and remaining <= 35):
            use_fast_eval = True
        eval_mode = "fast" if use_fast_eval else "full"

        # Logging every depth line can itself cause time losses under low clock.
        search_verbose_now = self.search_verbose and remaining > 20

        # Adjust time and depth based on position complexity.
        if simple:
            simple_scale = 0.72 if remaining <= 20 else 0.78
            time_for_move *= simple_scale
            LOGGER.info(
                "Game %s | simple position, fast line search (%.2fs, depth %d)",
                ctx.game_id,
                time_for_move,
                max_depth,
            )
        else:
            complex_scale = 0.90 if remaining <= 20 else 1.15
            time_for_move *= complex_scale
            LOGGER.info(
                "Game %s | complex position, deeper analysis (%.2fs, depth %d)",
                ctx.game_id,
                time_for_move,
                max_depth,
            )

        # Final hard safety pass: keep profile+reserve constraints after complexity scaling.
        final_clock_cap = self._profile_clock_cap(profile, remaining, increment)
        reserve_floor = self._reserve_floor(profile, increment, seen_moves_now)
        if remaining > reserve_floor:
            if profile == "bullet":
                reserve_cap = (remaining - reserve_floor) * 0.10
            elif profile == "blitz":
                reserve_cap = (remaining - reserve_floor) * 0.14
            elif profile == "rapid":
                reserve_cap = (remaining - reserve_floor) * 0.17
            else:
                reserve_cap = (remaining - reserve_floor) * 0.20
        else:
            reserve_cap = max(0.04, 0.05 + 0.18 * increment)
        hard_cap = self._hard_think_cap(remaining, increment)
        min_budget = 0.04 if profile == "bullet" else 0.05
        time_for_move = max(min_budget, min(time_for_move, final_clock_cap, reserve_cap, hard_cap))

        LOGGER.info(
            "Game %s | thinking time=%.2fs | depth cap=%d | remaining=%.2fs | profile=%s | tactical_swing=%d | eval=%s | search_verbose=%s",
            ctx.game_id,
            time_for_move,
            max_depth,
            remaining,
            profile,
            tactical_swing,
            eval_mode,
            search_verbose_now,
        )

        # Keep a small transport buffer so we can always submit before flag.
        if remaining <= 5:
            send_buffer = 0.14
        elif remaining <= 10:
            send_buffer = 0.12
        elif profile == "blitz" and remaining <= 30:
            send_buffer = 0.11
        elif remaining > 180:
            send_buffer = 0.20
        elif remaining > 30:
            send_buffer = 0.10
        elif remaining > 10:
            send_buffer = 0.08
        else:
            send_buffer = 0.04
        engine_time_limit = max(0.02, time_for_move - send_buffer)

        # ✅ SINGLE engine call per move
        try:
            with self.engine_lock:
                previous_eval_mode = getattr(engine.ENGINE, "eval_mode", "full")
                if previous_eval_mode != eval_mode:
                    engine.ENGINE.set_eval_mode(eval_mode)
                try:
                    best_move = engine.find_best_move(
                        position_for_engine,
                        max_depth=max_depth,
                        time_limit=engine_time_limit,
                        verbose=search_verbose_now,
                    )
                finally:
                    if previous_eval_mode != eval_mode:
                        engine.ENGINE.set_eval_mode(previous_eval_mode)
        except Exception:
            LOGGER.exception("Game %s | engine failed to provide a move.", ctx.game_id)
            return

        if best_move is None:
            with ctx.lock:
                checkmated = ctx.board.is_checkmate() and ctx.board.turn == ctx.our_color
            if checkmated:
                LOGGER.info("Game %s | no legal moves in checkmate, resigning.", ctx.game_id)
                self._safe_resign(ctx)
            else:
                LOGGER.error("Game %s | engine returned no move unexpectedly.", ctx.game_id)
            return

        with ctx.lock:
            if ctx.finished or ctx.our_color is None or not is_our_turn(ctx.board, ctx.our_color):
                return

            if best_move not in ctx.board.legal_moves:
                LOGGER.error("Game %s | engine produced illegal move %s", ctx.game_id, best_move.uci())
                fallback_move = self._deterministic_fallback_move(ctx.board.copy(stack=True))
                if fallback_move is None:
                    LOGGER.error("Game %s | no fallback legal move available.", ctx.game_id)
                    if ctx.board.is_checkmate() and ctx.board.turn == ctx.our_color:
                        should_resign = True
                    else:
                        should_resign = False
                else:
                    best_move = fallback_move
                    should_resign = False
                    LOGGER.warning("Game %s | using deterministic fallback move %s", ctx.game_id, best_move.uci())
            else:
                should_resign = False

            if should_resign:
                move_uci = ""
            else:
                move_uci = best_move.uci()
                try:
                    apply_move(ctx.board, move_uci)
                    ctx.seen_moves.append(move_uci)
                except Exception:
                    LOGGER.exception("Game %s | failed to apply local engine move %s", ctx.game_id, move_uci)
                    ctx.needs_resync = True
                    return

        if should_resign:
            self._safe_resign(ctx)
            return

        sent = False
        for attempt in range(1, 3):
            try:
                self.adapter.make_move(ctx.game_id, move_uci)
                sent = True
                break
            except Exception:
                if attempt < 2:
                    LOGGER.warning(
                        "Game %s | move send failed (attempt %d), retrying...",
                        ctx.game_id,
                        attempt,
                    )
                    time.sleep(0.12 * attempt)
                else:
                    LOGGER.exception("Game %s | failed to send move %s after retries", ctx.game_id, move_uci)

        if sent:
            LOGGER.info("Game %s | engine move: %s", ctx.game_id, move_uci)
            return

        with ctx.lock:
            # Roll back local optimistic update and force a server-state rebuild on next event.
            if ctx.seen_moves and ctx.seen_moves[-1] == move_uci:
                ctx.seen_moves.pop()
                if ctx.board.move_stack and ctx.board.peek().uci() == move_uci:
                    ctx.board.pop()
            ctx.needs_resync = True
            LOGGER.warning("Game %s | rolled back unsent move %s and marked resync", ctx.game_id, move_uci)

    def _safe_resign(self, ctx: GameContext):
        with ctx.lock:
            if ctx.finished:
                return
            ctx.finished = True
        try:
            self.adapter.resign_game(ctx.game_id)
            LOGGER.info("Game %s | resigned.", ctx.game_id)
        except Exception:
            LOGGER.exception("Game %s | resign request failed.", ctx.game_id)


def main():
    parser = argparse.ArgumentParser(description="Run the Lichess bot")
    parser.add_argument(
        "--search-verbose",
        action="store_true",
        help="Print iterative search lines (depth/score/nodes) during bot games.",
    )
    parser.add_argument(
        "--full-eval-complex",
        action="store_true",
        help="Disable fast eval mode in complex middlegames.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
    )

    adapter = LichessAdapter.from_env(token_env="LICHESS_TOKEN", logger=LOGGER)
    runner = LichessBotRunner(
        adapter=adapter,
        search_verbose=args.search_verbose,
        use_fast_eval_in_complex=not args.full_eval_complex,
    )
    LOGGER.info(
        "Bot config | search_verbose=%s | fast_eval_in_complex=%s",
        args.search_verbose,
        not args.full_eval_complex,
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested by user.")
    finally:
        runner.stop()


if __name__ == "__main__":
    main()