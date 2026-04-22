#!/usr/bin/env python3
"""
Stockfish-referenced blunder audit with live progress output.

Purpose:
- Quantify move-quality loss (centipawn loss) on real positions.
- Distinguish "engine is unstable/buggy" from "engine is stable but plays weak moves".
"""

from __future__ import annotations

import argparse
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


def score_to_cp(score: chess.engine.PovScore, pov: chess.Color) -> int:
    # Large mate score mapping keeps ranking monotonic for our CPL metric.
    return score.pov(pov).score(mate_score=100000)


def main():
    parser = argparse.ArgumentParser(description="Stockfish blunder audit")
    parser.add_argument("--variant", default="baseline_identity", choices=sorted(PATCHES.keys()))
    parser.add_argument("--engine-source", default="engine.py")
    parser.add_argument("--pgn", default="tune_data.pgn")
    parser.add_argument("--positions", type=int, default=60)
    parser.add_argument("--min-ply", type=int, default=12)
    parser.add_argument("--max-ply", type=int, default=80)
    parser.add_argument("--engine-time", type=float, default=0.12)
    parser.add_argument("--engine-depth", type=int, default=6)
    parser.add_argument("--sf-depth", type=int, default=14)
    parser.add_argument("--max-tt-entries", type=int, default=120000)
    parser.add_argument("--stockfish", default=DEFAULT_STOCKFISH_PATH)
    args = parser.parse_args()

    pgn_path = Path(args.pgn)
    if not pgn_path.is_file():
        raise FileNotFoundError(f"PGN not found: {pgn_path}")

    if not Path(args.stockfish).is_file():
        raise FileNotFoundError(f"Stockfish binary not found: {args.stockfish}")

    engine_source = Path(args.engine_source)
    if not engine_source.is_file():
        raise FileNotFoundError(f"Engine source not found: {engine_source}")

    src_text = engine_source.read_text(encoding="utf-8")
    mod = build_module(src_text, f"engine_{args.variant}_audit", PATCHES[args.variant])
    eng = mod.Engine(max_tt_entries=args.max_tt_entries)
    if hasattr(eng, "set_book_enabled"):
        eng.set_book_enabled(False)

    fens = load_positions_from_pgn(
        pgn_path=pgn_path,
        max_positions=args.positions,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
    )

    if not fens:
        raise RuntimeError("No positions extracted from PGN for selected ply window")

    print("BLUNDER_AUDIT_START", flush=True)
    print(
        f"engine_source={engine_source.name} variant={args.variant} "
        f"positions={len(fens)} pgn={pgn_path.name} "
        f"engine_time={args.engine_time}s engine_depth={args.engine_depth} sf_depth={args.sf_depth}",
        flush=True,
    )

    exact_best = 0
    illegal = 0
    none_nonterminal = 0
    exceptions = 0
    cpl_values: List[int] = []
    mistake_100 = 0
    blunder_200 = 0
    severe_300 = 0

    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as sf:
        for i, fen in enumerate(fens, start=1):
            board = chess.Board(fen)
            side = board.turn

            try:
                mv = eng.find_best_move(
                    board,
                    max_depth=args.engine_depth,
                    time_limit=args.engine_time,
                    verbose=False,
                )
            except Exception:
                exceptions += 1
                print(f"pos {i}/{len(fens)} engine_exception", flush=True)
                continue

            if mv is None:
                if not board.is_game_over(claim_draw=True):
                    none_nonterminal += 1
                print(f"pos {i}/{len(fens)} move=None", flush=True)
                continue

            if mv not in board.legal_moves:
                illegal += 1
                print(f"pos {i}/{len(fens)} illegal_move={mv.uci()}", flush=True)
                continue

            info_best = sf.analyse(
                board,
                chess.engine.Limit(depth=args.sf_depth),
                info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
            )
            best_score = score_to_cp(info_best["score"], side)
            best_pv = info_best.get("pv") or []
            best_move = best_pv[0] if best_pv else None

            if best_move is not None and mv == best_move:
                exact_best += 1

            b2 = board.copy()
            b2.push(mv)
            info_after = sf.analyse(
                b2,
                chess.engine.Limit(depth=args.sf_depth),
                info=chess.engine.INFO_SCORE,
            )
            after_score = score_to_cp(info_after["score"], side)

            cpl = max(0, best_score - after_score)
            cpl_values.append(cpl)

            if cpl >= 100:
                mistake_100 += 1
            if cpl >= 200:
                blunder_200 += 1
            if cpl >= 300:
                severe_300 += 1

            print(
                f"pos {i}/{len(fens)} move={mv.uci()} best={best_move.uci() if best_move else 'none'} "
                f"cpl={cpl} cum_blunder200={blunder_200}",
                flush=True,
            )

    n = len(cpl_values)
    avg_cpl = statistics.fmean(cpl_values) if cpl_values else 0.0
    med_cpl = statistics.median(cpl_values) if cpl_values else 0.0

    print("BLUNDER_AUDIT_DONE", flush=True)
    print(
        f"summary variant={args.variant} positions_total={len(fens)} evaluated={n} "
        f"exact_best={exact_best} ({(100.0 * exact_best / max(1, n)):.1f}%) "
        f"avg_cpl={avg_cpl:.1f} median_cpl={med_cpl:.1f} "
        f"mistake100={mistake_100} ({(100.0 * mistake_100 / max(1, n)):.1f}%) "
        f"blunder200={blunder_200} ({(100.0 * blunder_200 / max(1, n)):.1f}%) "
        f"severe300={severe_300} ({(100.0 * severe_300 / max(1, n)):.1f}%) "
        f"illegal={illegal} none_nonterminal={none_nonterminal} exceptions={exceptions}",
        flush=True,
    )


if __name__ == "__main__":
    main()
