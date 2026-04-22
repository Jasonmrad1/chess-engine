#!/usr/bin/env python3
"""
Chunked self-play runner with hard per-game timeout and live progress output.

Why this exists:
- Avoid long opaque runs that appear stuck.
- Detect major bugs (illegal moves, None on non-terminal, exceptions).
- Contain potential infinite search behavior by killing a game process when it exceeds
  a configurable wall-clock timeout.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import chess


PATCHES: Dict[str, List[Tuple[str, str]]] = {
    # Identity variant for baseline-vs-baseline stability stress.
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
    # Runtime has recapture extension; this is used as capture-extension proxy.
    "disable_capture_extension_proxy_recapture": [
        (
            "if not ext and is_cap and prev_move and prev_was_capture:",
            "if False and not ext and is_cap and prev_move and prev_was_capture:",
        )
    ],
    # Runtime continuation history is 1-ply.
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

OPENINGS: List[str] = [
    chess.STARTING_FEN,
    "rnbqkb1r/pppp1ppp/5n2/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3",
    "rnbqk2r/pppp1ppp/4pn2/8/1b1PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 2 5",
    "r2q1rk1/pp2bppp/2np1n2/2p1p3/2P1P3/2NP1N2/PPQ1BPPP/R1B2RK1 w - - 2 10",
    "2r2rk1/pp1b1ppp/2n1pn2/q1bp4/3P4/2NBPN2/PPQ2PPP/2RR2K1 w - - 2 13",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/3PP3/2P1BN2/PP3PPP/RN1QKB1R w KQ - 0 7",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 4 6",
    "r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1PN2/PPP2PPP/R2QKB1R w KQkq - 2 10",
    "2kr3r/ppp2ppp/2n1bn2/3qp3/3P4/2N1PN2/PPP2PPP/R2QKB1R w KQ - 4 11",
    "r1bq1rk1/pp1n1ppp/2pbpn2/8/2BP4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 9",
    "r4rk1/1pp1qppp/p1np1n2/4p3/2BPP3/2N2N2/PPP1QPPP/2KR3R w - - 0 12",
]


@dataclass
class Totals:
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: float = 0.0
    illegal: int = 0
    none_nonterminal: int = 0
    exceptions: int = 0
    hangs: int = 0
    truncated: int = 0
    hard_timeouts: int = 0
    plies: int = 0


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


def make_engine(mod, max_tt_entries: int):
    e = mod.Engine(max_tt_entries=max_tt_entries)
    if hasattr(e, "set_book_enabled"):
        try:
            e.set_book_enabled(False)
        except Exception:
            pass
    return e


def game_worker(
    q: mp.Queue,
    src_text: str,
    patches: List[Tuple[str, str]],
    fen: str,
    variant_is_white: bool,
    max_tt_entries: int,
    max_depth: int,
    time_limit: float,
    max_plies: int,
    hang_threshold_s: float,
):
    try:
        base_mod = build_module(src_text, "engine_base_worker", [])
        var_mod = build_module(src_text, "engine_var_worker", patches)

        white = make_engine(var_mod if variant_is_white else base_mod, max_tt_entries)
        black = make_engine(base_mod if variant_is_white else var_mod, max_tt_entries)

        board = chess.Board(fen)
        illegal = none_nonterminal = exceptions = hangs = truncated = 0
        plies = 0

        for _ in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break

            eng = white if board.turn == chess.WHITE else black
            t0 = time.time()
            try:
                mv = eng.find_best_move(
                    board,
                    max_depth=max_depth,
                    time_limit=time_limit,
                    verbose=False,
                )
            except Exception:
                exceptions += 1
                break

            dt = time.time() - t0
            if dt > hang_threshold_s:
                hangs += 1

            if mv is None:
                if not board.is_game_over(claim_draw=True):
                    none_nonterminal += 1
                break

            if mv not in board.legal_moves:
                illegal += 1
                break

            board.push(mv)
            plies += 1

        if not board.is_game_over(claim_draw=True):
            truncated += 1

        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"

        if result == "1-0":
            points = 1.0 if variant_is_white else 0.0
        elif result == "0-1":
            points = 0.0 if variant_is_white else 1.0
        else:
            points = 0.5

        q.put(
            {
                "ok": True,
                "points": points,
                "result": result,
                "illegal": illegal,
                "none_nonterminal": none_nonterminal,
                "exceptions": exceptions,
                "hangs": hangs,
                "truncated": truncated,
                "plies": plies,
            }
        )

    except Exception:
        q.put({"ok": False, "trace": traceback.format_exc(limit=20)})


def run_one_game(
    src_text: str,
    patches: List[Tuple[str, str]],
    fen: str,
    variant_is_white: bool,
    args: argparse.Namespace,
):
    q: mp.Queue = mp.Queue()
    p = mp.Process(
        target=game_worker,
        args=(
            q,
            src_text,
            patches,
            fen,
            variant_is_white,
            args.max_tt_entries,
            args.depth,
            args.time_limit,
            args.max_plies,
            args.hang_threshold,
        ),
    )
    p.start()
    p.join(args.game_timeout)

    if p.is_alive():
        p.terminate()
        p.join(5)
        return {
            "ok": False,
            "hard_timeout": True,
            "points": 0.5,
            "result": "1/2-1/2",
            "illegal": 0,
            "none_nonterminal": 0,
            "exceptions": 0,
            "hangs": 0,
            "truncated": 1,
            "plies": 0,
            "trace": "hard_timeout",
        }

    if q.empty():
        return {
            "ok": False,
            "hard_timeout": False,
            "points": 0.5,
            "result": "1/2-1/2",
            "illegal": 0,
            "none_nonterminal": 0,
            "exceptions": 1,
            "hangs": 0,
            "truncated": 1,
            "plies": 0,
            "trace": "worker_exited_without_payload",
        }

    out = q.get()
    out["hard_timeout"] = False
    return out


def elo_from_p(p: float):
    if p <= 0.0 or p >= 1.0:
        return None
    return -400.0 * math.log10((1.0 / p) - 1.0)


def ci95(p: float, n: int):
    if n <= 0:
        return (0.0, 1.0)
    se = math.sqrt(max(0.0, p * (1.0 - p) / n))
    return (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))


def main():
    parser = argparse.ArgumentParser(description="Chunked self-play with hard game timeout")
    parser.add_argument("--variant", required=True, choices=sorted(PATCHES.keys()))
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each opening pair this many times")
    parser.add_argument("--openings", type=int, default=10, help="How many openings from the built-in list")
    parser.add_argument("--time-limit", type=float, default=0.10)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--game-timeout", type=float, default=90.0, help="Hard wall-clock timeout per game")
    parser.add_argument("--hang-threshold", type=float, default=2.5, help="Per-move soft overrun marker")
    parser.add_argument("--max-tt-entries", type=int, default=120_000)
    args = parser.parse_args()

    src_text = Path("engine.py").read_text(encoding="utf-8")
    patch_list = PATCHES[args.variant]

    if args.openings < 1:
        raise ValueError("--openings must be >= 1")

    openings = OPENINGS[: min(args.openings, len(OPENINGS))]
    total_games = len(openings) * 2 * args.repeats

    print("SELFPLAY_CHUNK_START", flush=True)
    print(
        f"variant={args.variant} games={total_games} openings={len(openings)} repeats={args.repeats} "
        f"tc={args.time_limit}s depth={args.depth} max_plies={args.max_plies} "
        f"game_timeout={args.game_timeout}s",
        flush=True,
    )

    totals = Totals()
    idx = 0

    for rep in range(args.repeats):
        for fen_idx, fen in enumerate(openings, start=1):
            for variant_is_white in (False, True):
                idx += 1
                out = run_one_game(src_text, patch_list, fen, variant_is_white, args)

                totals.games += 1
                totals.points += out["points"]
                if out["points"] == 1.0:
                    totals.wins += 1
                elif out["points"] == 0.5:
                    totals.draws += 1
                else:
                    totals.losses += 1

                totals.illegal += out["illegal"]
                totals.none_nonterminal += out["none_nonterminal"]
                totals.exceptions += out["exceptions"] + (0 if out.get("ok", True) else 0)
                totals.hangs += out["hangs"]
                totals.truncated += out["truncated"]
                totals.hard_timeouts += 1 if out.get("hard_timeout", False) else 0
                totals.plies += out["plies"]

                side = "W" if variant_is_white else "B"
                print(
                    f"progress {idx}/{total_games} rep={rep+1} fen={fen_idx}/{len(openings)} side={side} "
                    f"res={out.get('result','?')} pts={out['points']:.1f} "
                    f"cum={totals.points:.1f}/{totals.games} "
                    f"bugs(illegal={totals.illegal},none={totals.none_nonterminal},"
                    f"exc={totals.exceptions},hard_to={totals.hard_timeouts})",
                    flush=True,
                )

                if not out.get("ok", True):
                    print(f"worker_issue: {out.get('trace', 'unknown')}", flush=True)

    p = totals.points / totals.games if totals.games else 0.0
    elo = elo_from_p(p)
    lo, hi = ci95(p, totals.games)

    print("SELFPLAY_CHUNK_DONE", flush=True)
    print(
        f"summary variant={args.variant} score={totals.points:.1f}/{totals.games} ({100.0*p:.1f}%) "
        f"wdl={totals.wins}-{totals.draws}-{totals.losses} "
        f"elo={'n/a' if elo is None else f'{elo:+.1f}'} "
        f"ci95=[{100.0*lo:.1f},{100.0*hi:.1f}] "
        f"illegal={totals.illegal} none_nonterminal={totals.none_nonterminal} "
        f"exceptions={totals.exceptions} hangs={totals.hangs} hard_timeouts={totals.hard_timeouts} "
        f"truncated={totals.truncated} avg_plies={totals.plies / max(1, totals.games):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
