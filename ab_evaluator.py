from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import inspect
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chess
import chess.pgn


@dataclass(frozen=True)
class EvalConfig:
    total_games: int = 200
    time_per_move_ms: int = 100
    random_seed: Optional[int] = None
    result_log_path: Optional[Path] = None
    pgn_export_path: Optional[Path] = None
    max_plies_per_game: int = 512
    show_progress: bool = True

    def validate(self) -> None:
        if self.total_games < 20:
            raise ValueError("total_games must be at least 20")
        if self.time_per_move_ms <= 0:
            raise ValueError("time_per_move_ms must be positive")
        if self.max_plies_per_game <= 0:
            raise ValueError("max_plies_per_game must be positive")


@dataclass
class GameRecord:
    game_index: int
    white_name: str
    black_name: str
    result: str
    termination: str
    move_count: int
    fallback_moves: int
    moves_uci: list[str]


@dataclass
class MatchSummary:
    total_games: int
    engine_a_wins: int
    engine_b_wins: int
    draws: int
    engine_a_win_rate: float
    engine_b_win_rate: float
    draw_rate: float
    expected_score_a: float
    elo_diff_a_vs_b: float
    confidence: str


@dataclass
class EvaluationResult:
    summary: MatchSummary
    games: list[GameRecord]


class EngineAdapter:
    """Normalizes engine interfaces into a single select_move(board) API."""

    _TIME_SETTERS_MS = (
        "set_time_per_move_ms",
        "set_move_time_ms",
        "set_time_limit_ms",
    )
    _TIME_SETTERS_SECONDS = (
        "set_time_per_move",
        "set_move_time",
        "set_time_limit",
    )

    def __init__(self, engine: Any, name: str, time_per_move_ms: int):
        self.engine = engine
        self.name = name
        self.time_per_move_ms = time_per_move_ms
        self._move_function = self._resolve_move_function(engine)
        self._configure_engine_time()

    @staticmethod
    def _resolve_move_function(engine: Any):
        if hasattr(engine, "get_best_move") and callable(getattr(engine, "get_best_move")):
            return getattr(engine, "get_best_move")
        if hasattr(engine, "find_best_move") and callable(getattr(engine, "find_best_move")):
            return getattr(engine, "find_best_move")
        if callable(engine):
            return engine
        raise TypeError(
            "Engine must expose get_best_move(board), find_best_move(board), or be callable(board)."
        )

    def _configure_engine_time(self) -> None:
        seconds = self.time_per_move_ms / 1000.0

        for method_name in self._TIME_SETTERS_MS:
            method = getattr(self.engine, method_name, None)
            if callable(method):
                method(self.time_per_move_ms)
                return

        for method_name in self._TIME_SETTERS_SECONDS:
            method = getattr(self.engine, method_name, None)
            if callable(method):
                method(seconds)
                return

    def _call_move_function(self, board: chess.Board) -> Any:
        fn = self._move_function
        seconds = self.time_per_move_ms / 1000.0

        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn(board)

        parameter_names = signature.parameters.keys()
        kwargs: dict[str, Any] = {}

        if "time_limit_ms" in parameter_names:
            kwargs["time_limit_ms"] = self.time_per_move_ms
        if "move_time_ms" in parameter_names:
            kwargs["move_time_ms"] = self.time_per_move_ms
        if "time_ms" in parameter_names:
            kwargs["time_ms"] = self.time_per_move_ms
        if "time_limit" in parameter_names:
            kwargs["time_limit"] = seconds
        if "move_time" in parameter_names:
            kwargs["move_time"] = seconds
        if "time_per_move" in parameter_names:
            kwargs["time_per_move"] = seconds
        if "verbose" in parameter_names:
            kwargs["verbose"] = False

        return fn(board, **kwargs)

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        raw_move = self._call_move_function(board)
        return normalize_move(raw_move, board)


def normalize_move(raw_move: Any, board: chess.Board) -> Optional[chess.Move]:
    if raw_move is None:
        return None

    if isinstance(raw_move, chess.Move):
        move = raw_move
    elif isinstance(raw_move, str):
        try:
            move = chess.Move.from_uci(raw_move)
        except ValueError:
            return None
    else:
        return None

    if move not in board.legal_moves:
        return None
    return move


def pick_fallback_move(board: chess.Board, rng: Optional[random.Random]) -> Optional[chess.Move]:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    if rng is None:
        legal_moves.sort(key=lambda m: m.uci())
        return legal_moves[0]

    return rng.choice(legal_moves)


def play_single_game(
    game_index: int,
    white_engine: EngineAdapter,
    black_engine: EngineAdapter,
    max_plies_per_game: int,
    rng: Optional[random.Random],
) -> GameRecord:
    board = chess.Board()
    moves_uci: list[str] = []
    fallback_moves = 0
    termination = "normal"

    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) >= max_plies_per_game:
            termination = "max_plies_limit"
            break

        mover = white_engine if board.turn == chess.WHITE else black_engine
        move = mover.select_move(board)

        if move is None:
            fallback_moves += 1
            move = pick_fallback_move(board, rng)
            if move is None:
                break

        moves_uci.append(move.uci())
        board.push(move)

    if termination == "max_plies_limit":
        result = "1/2-1/2"
    else:
        result = board.result(claim_draw=True)
        if result == "*":
            result = "1/2-1/2"

        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            termination = outcome.termination.name.lower()
        elif termination == "normal":
            termination = "unknown"

    return GameRecord(
        game_index=game_index,
        white_name=white_engine.name,
        black_name=black_engine.name,
        result=result,
        termination=termination,
        move_count=len(moves_uci),
        fallback_moves=fallback_moves,
        moves_uci=moves_uci,
    )


def score_record_for_engine_a(record: GameRecord, engine_a_is_white: bool) -> tuple[int, int, int]:
    if record.result == "1-0":
        if engine_a_is_white:
            return 1, 0, 0
        return 0, 1, 0

    if record.result == "0-1":
        if engine_a_is_white:
            return 0, 1, 0
        return 1, 0, 0

    return 0, 0, 1


def confidence_from_game_count(total_games: int) -> str:
    if total_games < 200:
        return "low"
    if total_games < 500:
        return "medium"
    return "high"


def elo_from_expected_score(expected_score: float) -> float:
    if expected_score <= 0.0:
        return float("-inf")
    if expected_score >= 1.0:
        return float("inf")
    return 400.0 * math.log10(expected_score / (1.0 - expected_score))


def _write_pgn_game(output_handle, record: GameRecord) -> None:
    game = chess.pgn.Game()
    game.headers["Event"] = "Engine A/B Evaluation"
    game.headers["Site"] = "Local"
    game.headers["Date"] = dt.date.today().strftime("%Y.%m.%d")
    game.headers["Round"] = str(record.game_index)
    game.headers["White"] = record.white_name
    game.headers["Black"] = record.black_name
    game.headers["Result"] = record.result
    game.headers["Termination"] = record.termination

    node = game
    for uci in record.moves_uci:
        node = node.add_variation(chess.Move.from_uci(uci))

    print(game, file=output_handle, end="\n\n")


def run_ab_evaluation(
    engine_a: Any,
    engine_b: Any,
    config: EvalConfig,
    engine_a_name: str = "Engine A",
    engine_b_name: str = "Engine B",
) -> EvaluationResult:
    config.validate()

    if config.random_seed is None:
        rng = None
    else:
        random.seed(config.random_seed)
        rng = random.Random(config.random_seed)

    adapter_a = EngineAdapter(engine_a, engine_a_name, config.time_per_move_ms)
    adapter_b = EngineAdapter(engine_b, engine_b_name, config.time_per_move_ms)

    game_records: list[GameRecord] = []
    a_wins = 0
    b_wins = 0
    draws = 0

    log_handle = None
    pgn_handle = None
    csv_writer = None

    try:
        if config.result_log_path is not None:
            config.result_log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = config.result_log_path.open("w", newline="", encoding="utf-8")
            csv_writer = csv.writer(log_handle)
            csv_writer.writerow(
                [
                    "game",
                    "white",
                    "black",
                    "result",
                    "termination",
                    "moves",
                    "fallback_moves",
                ]
            )

        if config.pgn_export_path is not None:
            config.pgn_export_path.parent.mkdir(parents=True, exist_ok=True)
            pgn_handle = config.pgn_export_path.open("w", encoding="utf-8")

        for game_index in range(1, config.total_games + 1):
            engine_a_is_white = (game_index % 2 == 1)

            white_engine = adapter_a if engine_a_is_white else adapter_b
            black_engine = adapter_b if engine_a_is_white else adapter_a

            record = play_single_game(
                game_index=game_index,
                white_engine=white_engine,
                black_engine=black_engine,
                max_plies_per_game=config.max_plies_per_game,
                rng=rng,
            )
            game_records.append(record)

            aw, bw, dr = score_record_for_engine_a(record, engine_a_is_white)
            a_wins += aw
            b_wins += bw
            draws += dr

            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        record.game_index,
                        record.white_name,
                        record.black_name,
                        record.result,
                        record.termination,
                        record.move_count,
                        record.fallback_moves,
                    ]
                )
                log_handle.flush()

            if pgn_handle is not None:
                _write_pgn_game(pgn_handle, record)
                pgn_handle.flush()

            if config.show_progress and (game_index % 10 == 0 or game_index == config.total_games):
                print(
                    f"[{game_index}/{config.total_games}] "
                    f"A wins: {a_wins}, B wins: {b_wins}, Draws: {draws}"
                )

    finally:
        if log_handle is not None:
            log_handle.close()
        if pgn_handle is not None:
            pgn_handle.close()

    total_games = len(game_records)
    expected_score_a = (a_wins + 0.5 * draws) / total_games
    elo_diff = elo_from_expected_score(expected_score_a)

    summary = MatchSummary(
        total_games=total_games,
        engine_a_wins=a_wins,
        engine_b_wins=b_wins,
        draws=draws,
        engine_a_win_rate=a_wins / total_games,
        engine_b_win_rate=b_wins / total_games,
        draw_rate=draws / total_games,
        expected_score_a=expected_score_a,
        elo_diff_a_vs_b=elo_diff,
        confidence=confidence_from_game_count(total_games),
    )

    return EvaluationResult(summary=summary, games=game_records)


def format_elo(elo: float) -> str:
    if math.isinf(elo):
        return "+inf" if elo > 0 else "-inf"
    return f"{elo:+.2f}"


def print_summary(summary: MatchSummary, engine_a_name: str, engine_b_name: str) -> None:
    print("\n=== A/B Evaluation Summary ===")
    print(f"Total games: {summary.total_games}")
    print(
        f"Score: {engine_a_name} wins {summary.engine_a_wins} | "
        f"{engine_b_name} wins {summary.engine_b_wins} | Draws {summary.draws}"
    )
    print(
        f"Win rate: {engine_a_name} {summary.engine_a_win_rate * 100:.2f}% | "
        f"{engine_b_name} {summary.engine_b_win_rate * 100:.2f}%"
    )
    print(f"Draw rate: {summary.draw_rate * 100:.2f}%")
    print(f"Expected score ({engine_a_name}): {summary.expected_score_a:.4f}")
    print(f"Estimated Elo difference ({engine_a_name} - {engine_b_name}): {format_elo(summary.elo_diff_a_vs_b)}")
    print(f"Confidence: {summary.confidence}")


def _split_spec(spec: str) -> tuple[str, Optional[str]]:
    if ":" not in spec:
        return spec, None
    module_name, attr_name = spec.split(":", 1)
    if not module_name:
        raise ValueError(f"Invalid engine spec '{spec}': missing module name")
    return module_name, attr_name or None


def _try_call_zero_arg_factory(target: Any) -> Any:
    if not callable(target):
        return target

    if hasattr(target, "get_best_move") or hasattr(target, "find_best_move"):
        return target

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return target

    required = [
        p
        for p in signature.parameters.values()
        if p.default is inspect._empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if required:
        return target

    created = target()
    if created is None:
        raise ValueError("Engine factory returned None")
    return created


def load_engine_from_spec(spec: str) -> Any:
    """
    Load an engine object from an import spec.

    Supported examples:
      module:ENGINE
      module:EngineClass
      module:build_engine
      module:get_best_move
    """
    module_name, attr_name = _split_spec(spec)
    module = importlib.import_module(module_name)

    if attr_name is None:
        if hasattr(module, "ENGINE"):
            target = getattr(module, "ENGINE")
        elif hasattr(module, "get_best_move"):
            target = getattr(module, "get_best_move")
        elif hasattr(module, "find_best_move"):
            target = getattr(module, "find_best_move")
        else:
            raise ValueError(
                f"No default engine target in module '{module_name}'. "
                "Provide an explicit spec like module:EngineClass or module:ENGINE."
            )
    else:
        target = module
        for part in attr_name.split("."):
            if not hasattr(target, part):
                raise ValueError(f"Could not resolve '{part}' in spec '{spec}'")
            target = getattr(target, part)

    if inspect.isclass(target):
        return target()

    return _try_call_zero_arg_factory(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Engine A vs Engine B via self-play with alternating colors, "
            "fixed move time, Elo estimate, and optional logs."
        )
    )
    parser.add_argument("--engine-a", required=True, help="Engine A import spec (e.g. engine_old:ENGINE)")
    parser.add_argument("--engine-b", required=True, help="Engine B import spec (e.g. engine_new:ENGINE)")
    parser.add_argument("--name-a", default="Engine A", help="Display name for Engine A")
    parser.add_argument("--name-b", default="Engine B", help="Display name for Engine B")
    parser.add_argument("--games", type=int, default=200, help="Total number of games (>=200)")
    parser.add_argument("--time-ms", type=int, default=100, help="Fixed time per move in milliseconds")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible fallback move ordering")
    parser.add_argument("--log-file", default=None, help="Optional CSV game result log path")
    parser.add_argument("--pgn-file", default=None, help="Optional PGN export path")
    parser.add_argument("--max-plies", type=int, default=512, help="Safety cap on half-moves per game")
    parser.add_argument("--quiet", action="store_true", help="Disable periodic progress output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = EvalConfig(
        total_games=args.games,
        time_per_move_ms=args.time_ms,
        random_seed=args.seed,
        result_log_path=Path(args.log_file) if args.log_file else None,
        pgn_export_path=Path(args.pgn_file) if args.pgn_file else None,
        max_plies_per_game=args.max_plies,
        show_progress=not args.quiet,
    )

    engine_a = load_engine_from_spec(args.engine_a)
    engine_b = load_engine_from_spec(args.engine_b)

    result = run_ab_evaluation(
        engine_a=engine_a,
        engine_b=engine_b,
        config=config,
        engine_a_name=args.name_a,
        engine_b_name=args.name_b,
    )

    print_summary(result.summary, args.name_a, args.name_b)

    if config.result_log_path is not None:
        print(f"Game result log written to: {config.result_log_path}")
    if config.pgn_export_path is not None:
        print(f"PGN export written to: {config.pgn_export_path}")


if __name__ == "__main__":
    main()
