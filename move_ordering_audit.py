#!/usr/bin/env python3
"""
Move-ordering quality audit against Stockfish best moves.

Measures whether internal ordering ranks the Stockfish best move near the front.
"""

from __future__ import annotations

import argparse
import inspect
import statistics
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn

DEFAULT_STOCKFISH_PATH = r"C:\CS\Stockfish\stockfish-windows-x86-64-avx2.exe"

PATCHES: Dict[str, List[Tuple[str, str]]] = {
    "baseline_identity": [],
    "disable_probcut": [
        (
            "if (not pv_node and depth >= 7 and not in_check and abs(beta) < MATE_BOUND):",
            "if False and (not pv_node and depth >= 7 and not in_check and abs(beta) < MATE_BOUND):",
        )
    ],
    "disable_multicut": [
        (
            "if fail_highs >= 3 and not pv_node and depth >= 4:",
            "if False and fail_highs >= 3 and not pv_node and depth >= 4:",
        )
    ],
    "disable_extended_futility_depth4": [
        (
            "and new_depth <= 4 and move_count > 0",
            "and new_depth <= 3 and move_count > 0",
        )
    ],
    "disable_threat_extension": [
        (
            "if not ext and not is_cap and ply <= 4:",
            "if False and not ext and not is_cap and ply <= 4:",
        )
    ],
    "disable_capture_extension_proxy_recapture": [
        (
            "if not ext and is_cap and prev_move and prev_was_capture:",
            "if False and not ext and is_cap and prev_move and prev_was_capture:",
        )
    ],
    "disable_cont_history_1ply": [
        (
            "        if prev_move:\n            ch = self.cont_hist.get(\n                (prev_move.from_square, prev_move.to_square,\n                 board.turn, move.from_square, move.to_square), 0)\n            s += min(600_000, ch) // 8\n",
            "        if False and prev_move:\n            ch = self.cont_hist.get(\n                (prev_move.from_square, prev_move.to_square,\n                 board.turn, move.from_square, move.to_square), 0)\n            s += min(600_000, ch) // 8\n",
        ),
        (
            "                    if prev_move:\n                        ck = (prev_move.from_square, prev_move.to_square,\n                              board.turn, move.from_square, move.to_square)\n                        self.cont_hist[ck] = self.cont_hist.get(ck, 0) + depth * depth\n",
            "                    if False and prev_move:\n                        ck = (prev_move.from_square, prev_move.to_square,\n                              board.turn, move.from_square, move.to_square)\n                        self.cont_hist[ck] = self.cont_hist.get(ck, 0) + depth * depth\n",
        ),
    ],
    "disable_combined_week4_applicable": [
        (
            "if (not pv_node and depth >= 7 and not in_check and abs(beta) < MATE_BOUND):",
            "if False and (not pv_node and depth >= 7 and not in_check and abs(beta) < MATE_BOUND):",
        ),
        (
            "if fail_highs >= 3 and not pv_node and depth >= 4:",
            "if False and fail_highs >= 3 and not pv_node and depth >= 4:",
        ),
        (
            "and new_depth <= 4 and move_count > 0",
            "and new_depth <= 3 and move_count > 0",
        ),
        (
            "if not ext and not is_cap and ply <= 4:",
            "if False and not ext and not is_cap and ply <= 4:",
        ),
        (
            "        if prev_move:\n            ch = self.cont_hist.get(\n                (prev_move.from_square, prev_move.to_square,\n                 board.turn, move.from_square, move.to_square), 0)\n            s += min(600_000, ch) // 8\n",
            "        if False and prev_move:\n            ch = self.cont_hist.get(\n                (prev_move.from_square, prev_move.to_square,\n                 board.turn, move.from_square, move.to_square), 0)\n            s += min(600_000, ch) // 8\n",
        ),
        (
            "                    if prev_move:\n                        ck = (prev_move.from_square, prev_move.to_square,\n                              board.turn, move.from_square, move.to_square)\n                        self.cont_hist[ck] = self.cont_hist.get(ck, 0) + depth * depth\n",
            "                    if False and prev_move:\n                        ck = (prev_move.from_square, prev_move.to_square,\n                              board.turn, move.from_square, move.to_square)\n                        self.cont_hist[ck] = self.cont_hist.get(ck, 0) + depth * depth\n",
        ),
    ],
}


def build_module(src_text: str, name: str, patches: List[Tuple[str, str]]):
    text = src_text
    for old, new in patches:
        if old not in text:
            raise RuntimeError(f"Patch anchor missing for {name}: {old[:80]!r}")
        text = text.replace(old, new, 1)
    mod = types.ModuleType(name)
    mod.__file__ = f"<{name}>"
    sys.modules[name] = mod
    exec(compile(text, mod.__file__, "exec"), mod.__dict__)
    return mod


def load_positions_from_pgn(pgn_path: Path, max_positions: int, min_ply: int, max_ply: int) -> List[str]:
    out: List[str] = []
    with pgn_path.open("r", encoding="utf-8", errors="replace") as f:
        while len(out) < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for ply, move in enumerate(game.mainline_moves(), start=1):
                if min_ply <= ply <= max_ply and not board.is_game_over(claim_draw=True):
                    if board.legal_moves.count() > 1:
                        out.append(board.fen())
                        if len(out) >= max_positions:
                            break
                board.push(move)
            if len(out) >= max_positions:
                break
    return out


def rank_of(move: chess.Move, seq: List[chess.Move]) -> int:
    try:
        return seq.index(move) + 1
    except ValueError:
        return len(seq) + 1


def pct(part: int, total: int) -> float:
    return 100.0 * part / max(1, total)


def call_ordered_moves(engine, board: chess.Board, legal_moves: List[chess.Move],
                       root_like: bool) -> List[chess.Move]:
    """Call _ordered_moves with backward-compatible kwargs filtering."""
    fn = engine._ordered_moves
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())

    kwargs = {
        "board": board,
        "moves": legal_moves,
        "tt_move": None,
        "ply": 0 if root_like else 2,
        "captures_only": False,
        "prev_move": None,
        "pv_node": root_like,
        "full_sort": root_like,
    }

    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(**filtered)


def summarize(label: str, ranks: List[int], top1: int, top3: int, top5: int):
    if not ranks:
        return f"{label}: no data"
    med = statistics.median(ranks)
    avg = statistics.fmean(ranks)
    return (
        f"{label}: top1={top1}/{len(ranks)} ({pct(top1,len(ranks)):.1f}%) "
        f"top3={top3}/{len(ranks)} ({pct(top3,len(ranks)):.1f}%) "
        f"top5={top5}/{len(ranks)} ({pct(top5,len(ranks)):.1f}%) "
        f"avg_rank={avg:.2f} median_rank={med:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Move-ordering quality audit")
    parser.add_argument("--engine-source", default="engine.py")
    parser.add_argument("--variant", default="baseline_identity", choices=sorted(PATCHES.keys()))
    parser.add_argument("--pgn", default="tune_data.pgn")
    parser.add_argument("--positions", type=int, default=120)
    parser.add_argument("--min-ply", type=int, default=12)
    parser.add_argument("--max-ply", type=int, default=80)
    parser.add_argument("--sf-depth", type=int, default=14)
    parser.add_argument("--stockfish", default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--max-tt-entries", type=int, default=120000)
    args = parser.parse_args()

    pgn_path = Path(args.pgn)
    if not pgn_path.is_file():
        raise FileNotFoundError(f"PGN not found: {pgn_path}")

    sf_path = Path(args.stockfish)
    if not sf_path.is_file():
        raise FileNotFoundError(f"Stockfish binary not found: {sf_path}")

    engine_source = Path(args.engine_source)
    if not engine_source.is_file():
        raise FileNotFoundError(f"Engine source not found: {engine_source}")

    src_text = engine_source.read_text(encoding="utf-8")
    mod = build_module(src_text, f"engine_{args.variant}_ord", PATCHES[args.variant])
    eng = mod.Engine(max_tt_entries=args.max_tt_entries)
    if hasattr(eng, "set_book_enabled"):
        eng.set_book_enabled(False)

    if not hasattr(eng, "_ordered_moves"):
        raise RuntimeError("Engine has no _ordered_moves method to audit")

    fens = load_positions_from_pgn(
        pgn_path=pgn_path,
        max_positions=args.positions,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
    )
    if not fens:
        raise RuntimeError("No positions extracted from PGN for selected ply window")

    print("MOVE_ORDER_AUDIT_START", flush=True)
    print(
        f"engine_source={engine_source.name} variant={args.variant} "
        f"positions={len(fens)} sf_depth={args.sf_depth}",
        flush=True,
    )

    root_ranks: List[int] = []
    staged_ranks: List[int] = []
    root_top1 = root_top3 = root_top5 = 0
    staged_top1 = staged_top3 = staged_top5 = 0

    with chess.engine.SimpleEngine.popen_uci(str(sf_path)) as sf:
        for i, fen in enumerate(fens, start=1):
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)

            info = sf.analyse(
                board,
                chess.engine.Limit(depth=args.sf_depth),
                info=chess.engine.INFO_PV,
            )
            pv = info.get("pv") or []
            if not pv:
                print(f"pos {i}/{len(fens)} no_sf_pv", flush=True)
                continue
            sf_best = pv[0]

            # Root-like ordering (pv node, full sort)
            ordered_root = call_ordered_moves(
                eng, board, legal_moves, root_like=True
            )
            rr = rank_of(sf_best, ordered_root)
            root_ranks.append(rr)
            if rr == 1:
                root_top1 += 1
            if rr <= 3:
                root_top3 += 1
            if rr <= 5:
                root_top5 += 1

            # Staged non-PV ordering path
            ordered_staged = call_ordered_moves(
                eng, board, legal_moves, root_like=False
            )
            sr = rank_of(sf_best, ordered_staged)
            staged_ranks.append(sr)
            if sr == 1:
                staged_top1 += 1
            if sr <= 3:
                staged_top3 += 1
            if sr <= 5:
                staged_top5 += 1

            print(
                f"pos {i}/{len(fens)} sf_best={sf_best.uci()} root_rank={rr} staged_rank={sr}",
                flush=True,
            )

    print("MOVE_ORDER_AUDIT_DONE", flush=True)
    print(
        summarize("root_order", root_ranks, root_top1, root_top3, root_top5),
        flush=True,
    )
    print(
        summarize("staged_order", staged_ranks, staged_top1, staged_top3, staged_top5),
        flush=True,
    )


if __name__ == "__main__":
    main()
