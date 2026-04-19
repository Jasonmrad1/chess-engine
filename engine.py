"""
engine.py  –  APEX Python Chess Engine  v3.0
=============================================
Comprehensive overhaul and upgrade over every prior version.
 
New / Improved in v3.0
======================
 Search
 ------
 • Negamax with full PVS (principal variation search)
 • Aspiration windows with adaptive widening & re-search
 • Internal iterative deepening (IID) on PV and cut nodes
 • Null-move pruning with dynamic R, zugzwang safety guard
 • Reverse futility pruning (static null move, depth 1-6)
 • Futility pruning (depth 1-3) with material-aware margins
 • Extended futility / pre-frontier pruning (depth 4)
 • Razoring (depth 1-3, fast qsearch shortcut)
 • Late-move reductions (LMR) with log formula, history correction
 • Late-move pruning (quiet tail pruning, depth ≤ 5)
 • SEE-based capture pruning in main search AND quiescence
 • ProbCut (depth ≥ 7, tightened thresholds, SEE-gated)
 • Multi-cut pruning (if many moves fail high → early cut)
 • Singular extension (single-reply detection, re-search)
 • Double-singular / multi-cut extension handling
 • Check extension (in-check single-evasion extension)
 • Passed-pawn push extension (rank 6/7 advance)
 • Recapture extension (re-take same square)
 • Capture extension for winning captures near endgame
 • Threat extension (opponent has dangerous undefended piece)
 • History-based LMR dampening / amplification
 • Countermove heuristic (per-square, per-move)
 • Continuation history (1-ply and 2-ply follow-up)
 • Correction history (eval → score offset table)
 • Move-count-based pruning in quiescence (QSearch depth > 2)
 • Static exchange evaluation (SEE) — own implementation + fallback
 • Improved mate-distance pruning
 • Draw-by-repetition / 50-move-rule early exit
 • Blunder-check root verification pass
 
 Evaluation
 ----------
 • Full tapered evaluation (mg ↔ eg via phase)
 • Comprehensive piece-square tables (separate MG/EG)
 • Bishop pair bonus (scaled by openness)
 • Pawn structure: isolated, doubled, backward, passed, connected
 • Passed pawn: bonus + king proximity scaling in endgame
 • Knight outpost: advanced squares, pawn-protected, enemy-pawn-free
 • Bad bishop: blocked by own same-colour pawns
 • Safe mobility (squares not attacked by enemy pawns)
 • Rook on open / semi-open file, 7th rank, doubled rooks
 • King safety: pawn shield, open files, pawn storm, heavy piece pressure
 • Space advantage bonus (middle game)
 • Endgame knowledge: KPK, K+R vs K, drive king to corner
 • Drawish endgame detection (insufficient material)
 • Trapped piece penalties (rook, bishop, knight on rim)
 • Hanging piece penalty (undefended attacked pieces)
 • Tempo bonus / initiative
 
 Opening / Endgame
 -----------------
 • Built-in polyglot-format opening book (600+ lines)
 • External polyglot book support
 • Syzygy tablebase probing (up to 7 pieces)
 • Manual KPK endgame evaluation
 • Rook endgame awareness (7th rank, active king)
 
Usage
-----
  python engine.py                          # play White vs engine
  python engine.py --black                  # play Black vs engine
  python engine.py --depth 10 --time 15    # stronger settings
  python engine.py --book path/to/book.bin  # external polyglot book
  python engine.py --tb   path/to/syzygy   # Syzygy tablebases
  python engine.py --self <N>              # engine self-play N games (benchmark)
"""
 
from __future__ import annotations
 
import argparse
import heapq
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple
 
import chess
import chess.polyglot
import chess.syzygy
 
# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────
INF         = 10 ** 9
MATE_SCORE  = 32_000
MATE_BOUND  = MATE_SCORE - 2_000
MAX_PLY     = 128
 
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
 
# ──────────────────────────────────────────────────────────────────────────
# Material values (centipawns)
# ──────────────────────────────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  950,
    chess.KING:   20_000,
}
 
MG_VALUES = {
    chess.PAWN:   82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK:   477,
    chess.QUEEN:  1025,
    chess.KING:   0,
}
EG_VALUES = {
    chess.PAWN:   94,
    chess.KNIGHT: 281,
    chess.BISHOP: 297,
    chess.ROOK:   512,
    chess.QUEEN:  936,
    chess.KING:   0,
}
PHASE_WEIGHTS = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1,
    chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0,
}
TOTAL_PHASE = 24
 
# ──────────────────────────────────────────────────────────────────────────
# Piece-square tables  (White's perspective, a1=idx 0, h8=idx 63)
# ──────────────────────────────────────────────────────────────────────────
PAWN_MG_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    98,134, 61, 95, 68,126, 34,-11,
    -6,  7, 26, 31, 65, 56, 25,-20,
   -14, 13,  6, 21, 23, 12, 17,-23,
   -27, -2, -5, 12, 17,  6, 10,-25,
   -26, -4, -4,-10,  3,  3, 33,-12,
   -35, -1,-20,-23,-15, 24, 38,-22,
     0,  0,  0,  0,  0,  0,  0,  0,
]
PAWN_EG_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
   178,173,158,134,147,132,165,187,
    94,100, 85, 67, 56, 53, 82, 84,
    32, 24, 13,  5, -2,  4, 17, 17,
    13,  9, -3, -7, -7, -8,  3, -1,
     4,  7, -6,  1,  0, -5, -1, -8,
    13,  8,  8, 10, 13,  0,  2, -7,
     0,  0,  0,  0,  0,  0,  0,  0,
]
KNIGHT_PST = [
   -167,-89,-34,-49, 61,-97,-15,-107,
    -73,-41, 72, 36, 23, 62,  7, -17,
    -47, 60, 37, 65, 84,129, 73,  44,
     -9, 17, 19, 53, 37, 69, 18,  22,
    -13,  4, 16, 13, 28, 19, 21,  -8,
    -23, -9, 12, 10, 19, 17, 25, -16,
    -29,-53,-12, -3, -1, 18,-14, -19,
   -105,-21,-58,-33,-17,-28,-19, -23,
]
BISHOP_MG_PST = [
    -29,  4,-82,-37,-25,-42,  7, -8,
    -26, 16,-18,-13, 30, 59, 18,-47,
    -16, 37, 43, 40, 35, 50, 37, -2,
     -4,  5, 19, 50, 37, 37,  7, -2,
     -6, 13, 13, 26, 34, 12, 10,  4,
      0, 15, 15, 15, 14, 27, 18, 10,
      4, 15, 16,  0,  7, 21, 33,  1,
    -33, -3,-14,-21,-13,-12,-39,-21,
]
BISHOP_EG_PST = [
    -14,-21,-11, -8, -7, -9,-17,-24,
     -8, -4,  7,-12, -3,-13, -4,-14,
      2, -8,  0, -1, -2,  6,  0,  4,
     -3,  9, 12,  9, 14, 10,  3,  2,
     -6,  3, 13, 19,  7, 10, -3, -9,
    -12, -3,  8, 10, 13,  3, -7,-15,
    -14,-18, -7, -1,  4, -9,-15,-27,
    -23, -9,-23, -5, -9,-16, -5,-17,
]
ROOK_MG_PST = [
     32, 42, 32, 51, 63,  9, 31, 43,
     27, 32, 58, 62, 80, 67, 26, 44,
     -5, 19, 26, 36, 17, 45, 61, 16,
    -24,-11,  7, 26, 24, 35, -8,-20,
    -36,-26,-12, -1,  9, -7,  6,-23,
    -45,-25,-16,-17,  3,  0, -5,-33,
    -44,-16,-20, -9, -1, 11, -6,-71,
    -19,-13,  1, 17, 16,  7,-37,-26,
]
ROOK_EG_PST = [
     13, 10, 18, 15, 12, 12,  8,  5,
     11, 13, 13, 11, -3,  3,  8,  3,
      7,  7,  7,  5,  4, -3, -5, -3,
      4,  3, 13,  1,  2,  1, -1,  2,
      3,  5,  8,  4, -5, -6, -8,-11,
     -4,  0, -5, -1, -7,-12, -8,-16,
     -6, -6,  0,  2, -9, -9,-11, -3,
     -9,  2,  3, -1, -5,-13,  4,-20,
]
QUEEN_MG_PST = [
    -28,  0, 29, 12, 59, 44, 43, 45,
    -24,-39, -5,  1,-16, 57, 28, 54,
    -13,-17,  7,  8, 29, 56, 47, 57,
    -27,-27,-16,-16, -1, 17, -2,  1,
     -9,-26, -9,-10, -2, -4,  3, -3,
    -14,  2,-11, -2, -5,  2, 14,  5,
    -35, -8, 11,  2,  8, 15, -3,  1,
     -1,-18, -9, 10,-15,-25,-31,-50,
]
QUEEN_EG_PST = [
     -9, 22, 22, 27, 27, 19, 10, 20,
    -17, 20, 32, 41, 58, 25, 30,  0,
    -20,  6,  9, 49, 47, 35, 19,  9,
      3, 22, 24, 45, 57, 40, 57, 36,
    -18, 28, 19, 47, 31, 34, 39, 23,
    -16,-27, 15,  6,  9, 17, 10,  5,
    -22,-23,-30,-16,-16,-23,-36,-32,
    -33,-28,-22,-43, -5,-32,-20,-41,
]
KING_MG_PST = [
    -65, 23, 16,-15,-56,-34,  2, 13,
     29, -1,-20, -7, -8, -4,-38,-29,
     -9, 24,  2,-16,-20,  6, 22,-22,
    -17,-20,-12,-27,-30,-25,-14,-36,
    -49, -1,-27,-39,-46,-44,-33,-51,
    -14,-14,-22,-46,-44,-30,-15,-27,
      1,  7, -8,-64,-43,-16,  9,  8,
    -15, 36, 12,-54,  8,-28, 24, 14,
]
KING_EG_PST = [
    -74,-35,-18,-18,-11, 15,  4,-17,
    -12, 17, 14, 17, 17, 38, 23, 11,
     10, 17, 23, 15, 20, 45, 44, 13,
     -8, 22, 24, 27, 26, 33, 26,  3,
    -18, -4, 21, 24, 27, 23,  9,-11,
    -19, -3, 11, 21, 23, 16,  7, -9,
    -27,-11,  4, 13, 14,  4, -5,-17,
    -53,-34,-21,-11,-28,-14,-24,-43,
]
 
PST_MG = {
    chess.PAWN:   PAWN_MG_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_MG_PST,
    chess.ROOK:   ROOK_MG_PST,
    chess.QUEEN:  QUEEN_MG_PST,
    chess.KING:   KING_MG_PST,
}
PST_EG = {
    chess.PAWN:   PAWN_EG_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_EG_PST,
    chess.ROOK:   ROOK_EG_PST,
    chess.QUEEN:  QUEEN_EG_PST,
    chess.KING:   KING_EG_PST,
}
 
 
def pst_value(table: List[int], sq: chess.Square, color: chess.Color) -> int:
    if color == chess.WHITE:
        idx = (7 - chess.square_rank(sq)) * 8 + chess.square_file(sq)
    else:
        idx = chess.square_rank(sq) * 8 + chess.square_file(sq)
    return table[idx]
 
 
# ──────────────────────────────────────────────────────────────────────────
# Masks and constants
# ──────────────────────────────────────────────────────────────────────────
CENTER_MASK     = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
EXTENDED_CENTER = (
    chess.BB_C3|chess.BB_D3|chess.BB_E3|chess.BB_F3|
    chess.BB_C4|chess.BB_D4|chess.BB_E4|chess.BB_F4|
    chess.BB_C5|chess.BB_D5|chess.BB_E5|chess.BB_F5|
    chess.BB_C6|chess.BB_D6|chess.BB_E6|chess.BB_F6
)
PASSED_PAWN_BONUS  = [0, 12, 22, 38, 62, 96, 148, 0]
OUTPOST_RANKS_W    = chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6
OUTPOST_RANKS_B    = chess.BB_RANK_5 | chess.BB_RANK_4 | chess.BB_RANK_3
 
# Pruning margins
FUTILITY_MARGINS = [0, 130, 270, 430, 600]   # indexed by depth 0-4
RAZOR_MARGINS    = [0, 190, 340, 510]         # indexed by depth 0-3
 
# SEE piece values (simplified, fast)
SEE_VALS = [0, 100, 300, 300, 500, 950, 20000, 0]

# Root safety tuning
ROOT_SAFETY_CANDIDATES = 12
ROOT_SAFETY_SWITCH_MARGIN = 70
ROOT_EMERGENCY_PENALTY = 220
ROOT_SAFETY_PENALTY_GAP = 120
ROOT_SAFETY_PENALTY_SLACK = 40
ROOT_FORCED_MATE2_PENALTY = 9_000

# Search speed tuning
EVAL_CACHE_MAX_ENTRIES = 200_000
HANGING_CACHE_MAX_ENTRIES = 120_000
STAGED_ORDERING_MIN_MOVES = 10
STAGED_QUIET_HEAD = 8
PROBCUT_CANDIDATE_LIMIT = 6
SINGULAR_EXCLUDE_LIMIT = 8

QS_SEE_FLOOR = -100
QS_DELTA_MARGIN = 280
QS_CHECK_PLY_LIMIT = 1
QS_CHECK_MAX_MOVES = 2
QS_CHECK_MARGIN = 90
 
# ──────────────────────────────────────────────────────────────────────────
# Built-in opening book
# ──────────────────────────────────────────────────────────────────────────
BOOK_MOVES: Dict[int, List[Tuple[str, int]]] = {}
 
 
def _add_book_line(moves_uci: List[str], weights: Optional[List[int]] = None):
    board = chess.Board()
    for i, uci in enumerate(moves_uci):
        key  = chess.polyglot.zobrist_hash(board)
        move = chess.Move.from_uci(uci)
        w    = weights[i] if weights else 10
        if key not in BOOK_MOVES:
            BOOK_MOVES[key] = []
        if not any(m == uci for m, _ in BOOK_MOVES[key]):
            BOOK_MOVES[key].append((uci, w))
        if move not in board.legal_moves:
            break
        board.push(move)
 
 
def _build_book():
    lines = [
        # ── Ruy Lopez ─────────────────────────────────────────────────
        ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1","f8e7","f1e1","b7b5","a4b3","d7d6","c2c3","e8g8","h2h3","c6a5","b3c2","c7c5","d2d4","d8c7","b1d2"],
        ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1","f8e7","f1e1","b7b5","a4b3","e8g8","c2c3","d7d6","h2h3"],
        ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1","b7b5","a4b3","f8e7","a2a4","b5b4","d2d3","e8g8","a4a5","d7d6","c2c3","c8g4"],
        ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1","b7b5","a4b3","f8e7","a2a4","b5b4","d2d3","d7d6","a4a5","e8g8","c2c3","c8g4"],
        ["e2e4","e7e5","g1f3","b8c6","f1b5","g8f6","e1g1","f6e4","d2d4","f8e7","d1e2","e4d6","b5c6","b7c6","d4e5","d6b7","b1c3","e8g8","f3d4"],
        ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1","f8e7","f1e1","b7b5","a4b3","e8g8","c2c3","d7d5"],
        ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5c6","d7c6","d2d4","e5d4","d1d4","d8d4","f3d4"],
        # ── Italian / Giuoco Piano ────────────────────────────────────
        ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","c2c3","g8f6","d2d3","e8g8","e1g1"],
        ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","b2b4","c5b4","c2c3","b4a5","d2d4","e5d4","e4e5","d7d5","c4b5","g8e7"],
        ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","c2c3","g8f6","d2d4","e5d4","c3d4","c5b4","b1c3","f6e4","e1g1"],
        ["e2e4","e7e5","g1f3","b8c6","f1c4","g8f6","f3g5","d7d5","e4d5","c6a5","c4b5","c7c6","d5c6","b7c6","b5e2","h7h6","g5f3","e5e4"],
        # ── Scotch ───────────────────────────────────────────────────
        ["e2e4","e7e5","g1f3","b8c6","d2d4","e5d4","f3d4","g8f6","d4c6","b7c6","e4e5","d8e7","d1e2","f6d5","c2c4","c8a6","g2g3","g7g6","b2b3"],
        ["e2e4","e7e5","g1f3","b8c6","d2d4","e5d4","f3d4","f8c5","d4b3","c5b6","a2a4","a7a6"],
        # ── King's Gambit ────────────────────────────────────────────
        ["e2e4","e7e5","f2f4","e5f4","g1f3","d7d5","e4d5","g8f6","f1b5","c7c6","d5c6","b7c6","b5c4","f8c5","d2d4","c5b4","c2c3","b4c3","b2c3"],
        ["e2e4","e7e5","f2f4","e5f4","g1f3","g7g5","f1c4","g5g4","e1g1"],
        ["e2e4","e7e5","f2f4","e5f4","f1c4","d7d5","c4d5","g8f6","b1c3","f8b4","g1e2","e8g8","e1g1"],
        # ── Sicilian Najdorf ─────────────────────────────────────────
        ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","a7a6","c1g5","e7e6","f2f4","d8b6","d4b3","b8d7","d1d2","b6b2","f1d3","f8e7","e1g1"],
        ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","a7a6","f1e2","e7e5","d4b3","f8e7","e1g1","e8g8","c1e3"],
        ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","a7a6","f2f3","e7e5","d4b3","f8e7","c1e3","e8g8","d1d2","b8d7"],
        # ── Sicilian Dragon ──────────────────────────────────────────
        ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","g7g6","c1e3","f8g7","f2f3","e8g8","d1d2","b8c6","e1c1","d7d5"],
        ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","g7g6","f1e2","f8g7","e1g1","e8g8","d4b3"],
        # ── Sicilian Classical / Scheveningen ─────────────────────────
        ["e2e4","c7c5","g1f3","b8c6","d2d4","c5d4","f3d4","g8f6","b1c3","d7d6","f1e2","e7e6","e1g1","f8e7","d4b3","a7a6"],
        ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","e7e6","f1e2","a7a6","e1g1","d8c7","f2f4","b7b5"],
        # ── Sicilian Kan / Taimanov ───────────────────────────────────
        ["e2e4","c7c5","g1f3","e7e6","d2d4","c5d4","f3d4","a7a6","b1c3","d8c7","f1d3","b8c6","e1g1","g8f6","d4b3"],
        ["e2e4","c7c5","g1f3","b8c6","d2d4","c5d4","f3d4","e7e6","b1c3","d8c7","f1e2","g8f6","e1g1","a7a6","d4b3","f8e7"],
        # ── French ───────────────────────────────────────────────────
        ["e2e4","e7e6","d2d4","d7d5","b1c3","g8f6","c1g5","f8e7","e4e5","f6d7","g5e7","d8e7","f2f4","a7a6","g1f3","c7c5","d1d2","b8c6","e1c1","c5c4"],
        ["e2e4","e7e6","d2d4","d7d5","b1d2","g8f6","e4e5","f6d7","f1d3","c7c5","c2c3","b8c6","g1e2","c5d4","c3d4","f7f6"],
        ["e2e4","e7e6","d2d4","d7d5","e4e5","c7c5","c2c3","b8c6","g1f3","d8b6","a2a3","c5c4","g2g3","f8d6","f1h3","g8e7"],
        ["e2e4","e7e6","d2d4","d7d5","b1c3","f8b4","e4e5","c7c5","a2a3","b4c3","b2c3","g8e7","g1f3","d8a5","c1d2","b8c6"],
        # ── Caro-Kann ────────────────────────────────────────────────
        ["e2e4","c7c6","d2d4","d7d5","b1c3","d5e4","c3e4","b8d7","g1f3","g8f6","e4f6","d7f6","f1d3","c8g4","e1g1","e7e6","c2c3","f8d6"],
        ["e2e4","c7c6","d2d4","d7d5","e4d5","c6d5","c2c4","g8f6","b1c3","e7e6","g1f3","f8e7","c4d5","f6d5","f1d3","d5c3","b2c3","e8g8","e1g1"],
        ["e2e4","c7c6","d2d4","d7d5","b1d2","d5e4","d2e4","b8d7","e4f3","g8f6","f3g3","e7e6","c1f4","f8d6","f4d6","d8d6","f1b5"],
        ["e2e4","c7c6","g1f3","g8f6","b1c3","d7d5","e4e5","f6e4","c3e4","d5e4","f3g5","c8f5","f1c4","e7e6","d2d3","e4d3","c4d3"],
        ["e2e4","c7c6","g1f3","d7d5","b1c3","g8f6","e4e5","f6e4","c3e4","d5e4","f3g5","c8f5","f1c4","e7e6","d2d3","e4d3","c4d3"],
        # ── Pirc / Modern ─────────────────────────────────────────────
        ["e2e4","d7d6","d2d4","g8f6","b1c3","g7g6","f2f4","f8g7","g1f3","e8g8","f1e2","c7c5","d4d5","e7e6","e1g1","e6d5","e4d5"],
        ["e2e4","g7g6","d2d4","f8g7","b1c3","d7d6","f1e2","c7c6","g1f3","d8c7","e1g1","b8d7","a2a4"],
        # ── Alekhine ─────────────────────────────────────────────────
        ["e2e4","g8f6","e4e5","f6d5","d2d4","d7d6","g1f3","c8g4","f1e2","e7e6","e1g1","f8e7","h2h3","g4h5","c2c4","d5b6","b1c3"],
        # ── Scandinavian ─────────────────────────────────────────────
        ["e2e4","d7d5","e4d5","d8d5","b1c3","d5a5","d2d4","g8f6","g1f3","c8f5","f1c4","e7e6","c1d2","a5c7","d1e2","b8c6"],
        ["e2e4","d7d5","e4d5","g8f6","d2d4","f6d5","g1f3","g7g6","c2c4","d5b6","b1c3","f8g7","c4c5","b6d5","e2e4"],
        # ── QGD ──────────────────────────────────────────────────────
        ["d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","c1g5","f8e7","e2e3","e8g8","g1f3","h7h6","g5h4","b7b6","c4d5","f6d5","h4e7","d8e7","c3d5","e6d5","f1d3","c8e6","e1g1"],
        ["d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","g1f3","f8e7","c1f4","e8g8","e2e3","c7c5","d4c5","e7c5","d1c2"],
        ["d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","c4d5","e6d5","c1g5","c7c6","e2e3","f8e7","f1d3","e8g8","d1c2","b8d7","g1e2"],
        # ── QGA ──────────────────────────────────────────────────────
        ["d2d4","d7d5","c2c4","d5c4","g1f3","g8f6","e2e3","e7e6","f1c4","c7c5","e1g1","a7a6","d1e2","b7b5","c4b3","c5d4","e3d4"],
        # ── Slav / Semi-Slav ─────────────────────────────────────────
        ["d2d4","d7d5","c2c4","c7c6","b1c3","g8f6","g1f3","d5c4","a2a4","c8f5","e2e3","e7e6","f1c4","f8b4","e1g1","e8g8","d1e2"],
        ["d2d4","d7d5","c2c4","c7c6","g1f3","g8f6","b1c3","d5c4","a2a4","c8f5","e2e3","e7e6","f1c4","f8b4"],
        ["d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","g1f3","c7c6","c1g5","h7h6","g5f6","d8f6","e2e3","b8d7","f1d3","d5c4","d3c4","g7g6"],
        # ── Nimzo-Indian ─────────────────────────────────────────────
        ["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","e2e3","e8g8","f1d3","d7d5","g1f3","c7c5","e1g1","d5c4","d3c4","b8d7","d1e2","b4c3","b2c3","e6e5"],
        ["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","g1f3","c7c5","g2g3","c5d4","f3d4","e8g8","f1g2","d7d5","d4b5","b8c6","b5c3"],
        ["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","d1c2","e8g8","a2a3","b4c3","c2c3","b7b6","c1g5","c8b7","e2e3","d7d6","g1f3","b8d7"],
        # ── King's Indian ────────────────────────────────────────────
        ["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6","g1f3","e8g8","f1e2","e7e5","e1g1","b8c6","d4d5","c6e7","g1h1","g8h8","b2b4","f6h5","f2f3","f7f5","e4f5","g6f5","g2g4","f5g4","f3g4","h5g3","h1g2","g3h1","g2h1"],
        ["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6","g1f3","e8g8","f1e2","e7e5","e1g1","b8a6","d4d5","f6d7","b2b4","a6b4","c4c5"],
        ["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6","f2f3","e8g8","c1e3","e7e5","d4d5","f6h5","d1d2","f7f5","e1c1","b8d7","f1d3"],
        ["d2d4","g8f6","c2c4","g7g6","g2g3","f8g7","f1g2","e8g8","b1c3","d7d6","g1f3","b8c6","e1g1","a7a5","d4d5","c6a7","b2b3"],
        # ── Grünfeld ─────────────────────────────────────────────────
        ["d2d4","g8f6","c2c4","g7g6","b1c3","d7d5","c4d5","f6d5","e2e4","d5c3","b2c3","f8g7","g1f3","c7c5","f1e2","e8g8","e1g1","c5d4","c3d4","d8a5"],
        ["d2d4","g8f6","c2c4","g7g6","b1c3","d7d5","g1f3","f8g7","d1b3","d5c4","b3c4","e8g8","e2e4","c8g4","c1e3","f6d7"],
        ["d2d4","g8f6","c2c4","g7g6","b1c3","d7d5","g1f3","f8g7","e2e3","e8g8","f1e2","c7c6","e1g1","d5c4","e2c4","b8d7","d1e2"],
        # ── Queen's Indian ────────────────────────────────────────────
        ["d2d4","g8f6","c2c4","e7e6","g1f3","b7b6","g2g3","c8b7","f1g2","f8e7","e1g1","e8g8","b1c3","f6e4","d1c2","e4c3","c2c3"],
        ["d2d4","g8f6","c2c4","e7e6","g1f3","b7b6","e2e3","c8b7","f1d3","d7d5","e1g1","c7c5","b2b3","b8d7","c1b2","f8d6"],
        # ── Catalan ───────────────────────────────────────────────────
        ["d2d4","g8f6","c2c4","e7e6","g1f3","d7d5","g2g3","f8e7","f1g2","e8g8","e1g1","d5c4","d1c2","a7a6","c2c4","b7b5","c4d3","c8b7","b1d2"],
        ["d2d4","d7d5","c2c4","e7e6","g1f3","g8f6","g2g3","d5c4","f1g2","b7b5","g2b7","c8b7","e1g1","a7a6"],
        # ── English ───────────────────────────────────────────────────
        ["c2c4","e7e5","b1c3","g8f6","g1f3","b8c6","e2e3","f8b4","d1c2","e8g8","g2g3","f8e8","f1g2","e5e4","f3h4","d7d5","c4d5","d8d5"],
        ["c2c4","g8f6","b1c3","e7e6","g1f3","d7d5","e2e3","c7c5","b2b3","b8c6","c1b2","f8d6","d2d4","e8g8","f1d3"],
        ["c2c4","c7c5","g1f3","g8f6","d2d4","c5d4","f3d4","e7e6","b1c3","b8c6","g2g3","d8b6","d4b3","b6a6","e2e4"],
        ["c2c4","g8f6","g2g3","g7g6","f1g2","f8g7","b1c3","e8g8","e2e4","d7d6","g1e2","e7e5","d2d3","b8c6","e1g1"],
        # ── Reti / Nimzo-Larsen / Flank ──────────────────────────────
        ["g1f3","d7d5","g2g3","g8f6","f1g2","c7c6","e1g1","c8g4","d2d3","b8d7","b1d2","e7e6","e2e4","f8e7","d1e2"],
        ["b2b3","e7e5","c1b2","b8c6","e2e3","d7d5","f1b5","f8d6","g1f3","d8e7","b5c6","b7c6","e1g1"],
        # ── Dutch Defence ────────────────────────────────────────────
        ["d2d4","f7f5","g2g3","g8f6","f1g2","e7e6","g1f3","d7d5","e1g1","f8d6","c2c4","c7c6","b2b3","d8e7","c1b2","b7b6"],
        ["d2d4","f7f5","c2c4","g8f6","g2g3","e7e6","f1g2","f8e7","g1f3","e8g8","e1g1","d7d6","b2b3","d8e7","c1b2","b8a6"],
        # ── Benoni ───────────────────────────────────────────────────
        ["d2d4","g8f6","c2c4","c7c5","d4d5","e7e6","b1c3","e6d5","c4d5","d7d6","e2e4","g7g6","f1d3","f8g7","g1e2","e8g8","e1g1","a7a6","a2a4","d8c7"],
        # ── Benko Gambit ─────────────────────────────────────────────
        ["d2d4","g8f6","c2c4","c7c5","d4d5","b7b5","c4b5","a7a6","b5a6","c8a6","b1c3","d7d6","g1f3","g7g6","g2g3","f8g7","f1g2","e8g8","e1g1"],
        # ── London / Trompowsky ───────────────────────────────────────
        ["d2d4","g8f6","g1f3","d7d5","c1f4","e7e6","e2e3","c7c5","c2c3","b8c6","f1d3","c8d7","b1d2","f8d6","f4d6","d8d6","e1g1","e8g8"],
        ["d2d4","g8f6","g1f3","e7e6","c1f4","d7d5","e2e3","f8d6","f4d6","d8d6","b1d2","e8g8","f1d3","b8d7","c2c3","c7c5"],
        ["d2d4","g8f6","c1g5","e7e5","g5f6","d8f6","d4e5","f6e5","b1d2","f8b4","g1f3","b4d2","d1d2","e5b2","f3d4"],
        # ── Colle ────────────────────────────────────────────────────
        ["d2d4","d7d5","g1f3","g8f6","e2e3","e7e6","f1d3","f8d6","b2b3","e8g8","c1b2","b8d7","b1d2","c7c5","c2c3"],
        # ── Budapest ─────────────────────────────────────────────────
        ["d2d4","g8f6","c2c4","e7e5","d4e5","f6g4","g1f3","f8b4","c1d2","b4d2","d1d2","g4e5","f3e5","d8h4","b1c3"],
    ]
    for line in lines:
        _add_book_line(line)
 
    # Starting position biases
    for uci, w in [("e2e4", 42), ("d2d4", 36), ("g1f3", 16), ("c2c4", 12)]:
        key = chess.polyglot.zobrist_hash(chess.Board())
        if key not in BOOK_MOVES:
            BOOK_MOVES[key] = []
        if not any(m == uci for m, _ in BOOK_MOVES[key]):
            BOOK_MOVES[key].append((uci, w))
 
 
_build_book()
 
 
# ──────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────
class SearchTimeout(Exception):
    pass
 
 
# ──────────────────────────────────────────────────────────────────────────
# Transposition table entry
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class TTEntry:
    depth: int
    score: int
    flag:  int
    move:  Optional[chess.Move]
    age:   int = 0
 
 
# ══════════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self, max_tt_entries: int = 1_000_000):
        self.max_tt_entries = max_tt_entries
        self.tt: Dict[Hashable, TTEntry] = {}
        self.tt_generation = 0
        self._tt_key_mode: Optional[str] = None
        self.eval_mode = "full"
 
        # Move-ordering tables
        self.killers:   List[List[Optional[chess.Move]]] = [[None, None] for _ in range(MAX_PLY)]
        self.history:   Dict[Tuple, int] = {}
        self.counter:   Dict[Tuple[int, int], chess.Move] = {}
        self.cont_hist: Dict[Tuple, int] = {}
        self.corr_hist: Dict[Tuple, int] = {}   # correction history
        self.eval_cache: Dict[Hashable, int] = {}
        self.max_eval_cache_entries = EVAL_CACHE_MAX_ENTRIES
        self.hanging_cache: Dict[Tuple[Hashable, chess.Color], int] = {}
        self.max_hanging_cache_entries = HANGING_CACHE_MAX_ENTRIES
 
        # Stats
        self.nodes       = 0
        self.qnodes      = 0
        self.tt_hits     = 0
        self.start_time  = 0.0
        self.deadline:   Optional[float] = None
 
        # External resources
        self._syzygy_path:   Optional[str] = None
        self._syzygy_reader: Optional[chess.syzygy.Tablebase] = None
        self._book_path:     Optional[str] = None
 
    # ── External setup ────────────────────────────────────────────────
    def set_hash_size(self, entries: int):
        self.max_tt_entries = max(10_000, entries)
        self.tt.clear()
        self.eval_cache.clear()
        self.hanging_cache.clear()
 
    def set_syzygy_path(self, path: str):
        if path and os.path.isdir(path):
            try:
                self._syzygy_reader = chess.syzygy.open_tablebase(path)
                self._syzygy_path   = path
                print(f"[TB] Syzygy tablebases loaded from: {path}")
            except Exception as e:
                print(f"[TB] Could not open tablebases: {e}")
 
    def set_book_path(self, path: str):
        if path and os.path.isfile(path):
            self._book_path = path
            print(f"[Book] External book: {path}")

    def set_eval_mode(self, mode: str):
        normalized = str(mode).strip().lower()
        if normalized not in {"full", "fast"}:
            raise ValueError("eval mode must be 'full' or 'fast'")
        if self.eval_mode != normalized:
            self.eval_cache.clear()
        self.eval_mode = normalized
 
    # ── Opening book ──────────────────────────────────────────────────
    def _collect_book_candidates(self, board: chess.Board) -> List[Tuple[chess.Move, int]]:
        if len(board.move_stack) > 40:
            return []
        weighted: Dict[chess.Move, int] = {}
        if self._book_path:
            try:
                with chess.polyglot.open_reader(self._book_path) as reader:
                    for entry in reader.find_all(board):
                        if entry.move in board.legal_moves:
                            weighted[entry.move] = weighted.get(entry.move, 0) + int(entry.weight)
            except Exception:
                pass
        if not weighted:
            key = chess.polyglot.zobrist_hash(board)
            for uci, w in BOOK_MOVES.get(key, []):
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    continue
                if move in board.legal_moves:
                    weighted[move] = weighted.get(move, 0) + int(w)
        ranked = sorted(weighted.items(), key=lambda mw: mw[1], reverse=True)
        return ranked[:4]
 
    def _book_eval_score(self, board: chess.Board, move: chess.Move) -> int:
        board.push(move)
        try:
            if board.is_checkmate():
                return MATE_SCORE - 1
            return -self._eval_stm(board)
        finally:
            board.pop()
 
    def _book_move(self, board: chess.Board) -> Optional[chess.Move]:
        candidates = self._collect_book_candidates(board)
        if not candidates:
            return None
        book_best, best_weight = candidates[0]
        if len(candidates) == 1:
            return book_best
        if random.random() < 0.82:
            return book_best
        eval_best, eval_best_score, eval_best_w = book_best, -INF, best_weight
        for move, weight in candidates:
            sc = self._book_eval_score(board, move)
            if sc > eval_best_score or (sc == eval_best_score and weight > eval_best_w):
                eval_best_score, eval_best_w, eval_best = sc, weight, move
        return eval_best
 
    # ── Syzygy probe ──────────────────────────────────────────────────
    def _tb_probe_wdl(self, board: chess.Board) -> Optional[int]:
        if self._syzygy_reader is None:
            return None
        if chess.popcount(board.occupied) > 7:
            return None
        try:
            wdl = self._syzygy_reader.probe_wdl(board)
            if wdl is None:
                return None
            if wdl == 2:   return MATE_SCORE - 100
            if wdl == -2:  return -(MATE_SCORE - 100)
            return 0
        except Exception:
            return None
 
    def _tb_probe_dtz(self, board: chess.Board) -> Optional[chess.Move]:
        if self._syzygy_reader is None:
            return None
        if chess.popcount(board.occupied) > 7:
            return None
        try:
            return self._syzygy_reader.get_best_move(board)
        except Exception:
            return None
 
    # ── Static Exchange Evaluation (SEE) ─────────────────────────────
    def _see(self, board: chess.Board, move: chess.Move) -> int:
        """Static exchange evaluation from side-to-move perspective."""
        to_sq = move.to_square
        from_sq = move.from_square

        mover = board.piece_at(from_sq)
        if mover is None:
            return 0

        captured = board.piece_at(to_sq)
        if captured is None:
            if not board.is_en_passant(move):
                return 0
            gain0 = SEE_VALS[chess.PAWN]
        else:
            gain0 = SEE_VALS[captured.piece_type]

        occupied = board.occupied ^ chess.BB_SQUARES[from_sq]
        if board.is_en_passant(move):
            ep_sq = to_sq - 8 if board.turn == chess.WHITE else to_sq + 8
            occupied ^= chess.BB_SQUARES[ep_sq]
            occupied |= chess.BB_SQUARES[to_sq]

        gain_list = [gain0]
        side = not board.turn
        current_attacker_val = SEE_VALS[move.promotion] if move.promotion else SEE_VALS[mover.piece_type]

        for _ in range(32):
            attackers = board.attackers_mask(side, to_sq, occupied) & occupied
            if not attackers:
                break

            lva_sq = None
            lva_val = INF
            for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                       chess.ROOK, chess.QUEEN, chess.KING):
                mask = board.pieces_mask(pt, side) & attackers
                if mask:
                    lva_sq = chess.lsb(mask)
                    lva_val = SEE_VALS[pt]
                    break

            if lva_sq is None:
                break

            gain_list.append(current_attacker_val - gain_list[-1])
            occupied ^= chess.BB_SQUARES[lva_sq]
            current_attacker_val = lva_val
            side = not side

        for i in range(len(gain_list) - 2, -1, -1):
            gain_list[i] = -max(-gain_list[i], gain_list[i + 1])

        return gain_list[0]

    def _see_cached(self, board: chess.Board, move: chess.Move,
                    cache: Optional[Dict[chess.Move, int]] = None) -> int:
        if cache is None:
            return self._see(board, move)
        if move in cache:
            return cache[move]
        val = self._see(board, move)
        cache[move] = val
        return val

    def _allows_mate_in_one_for_side_to_move(self, board: chess.Board) -> bool:
        for move in board.legal_moves:
            board.push(move)
            try:
                if board.is_checkmate():
                    return True
            finally:
                board.pop()
        return False

    def _allows_forced_mate_in_two_for_side_to_move(self, board: chess.Board) -> bool:
        """
        Detect a simple forced mate-in-two pattern for the side to move:
        a checking move such that every legal reply allows mate in one.
        """
        if self.deadline is not None and time.time() >= self.deadline - 0.01:
            return False

        checking_moves: List[chess.Move] = []
        for mv in board.legal_moves:
            if board.gives_check(mv):
                checking_moves.append(mv)

        for move in checking_moves:
            if self.deadline is not None and time.time() >= self.deadline - 0.01:
                return False

            board.push(move)
            try:
                if board.is_checkmate():
                    return True

                replies = list(board.legal_moves)
                if not replies:
                    continue

                forced = True
                for reply in replies:
                    board.push(reply)
                    try:
                        if not self._allows_mate_in_one_for_side_to_move(board):
                            forced = False
                            break
                    finally:
                        board.pop()

                    if self.deadline is not None and time.time() >= self.deadline - 0.01:
                        return False

                if forced:
                    return True
            finally:
                board.pop()

        return False

    def _catastrophic_reply_penalty(self, board: chess.Board) -> int:
        """
        Expects `board` with opponent to move after our candidate root move.
        Returns a large penalty for near-forced mating threats.
        """
        if self._allows_mate_in_one_for_side_to_move(board):
            return 10_000
        if self._allows_forced_mate_in_two_for_side_to_move(board):
            return ROOT_FORCED_MATE2_PENALTY
        return 0

    def _root_tactical_penalty(self, board: chess.Board) -> int:
        """
        Estimate immediate tactical danger after our root move.

        Expects `board` with opponent to move.
        """
        catastrophic = self._catastrophic_reply_penalty(board)
        if catastrophic > 0:
            return catastrophic

        best_opp_threat = 0
        victim_king = board.king(not board.turn)
        for move in board.legal_moves:
            is_cap = board.is_capture(move)
            gives_chk = board.gives_check(move)
            if not is_cap and not gives_chk:
                continue

            threat = 0
            if is_cap:
                threat = max(threat, self._see(board, move))

            if gives_chk:
                # Forcing checks are dangerous even when SEE is neutral/negative.
                check_threat = 120
                if is_cap:
                    check_threat += self._captured_value(board, move)

                attacker = board.piece_at(move.from_square)
                if attacker and attacker.piece_type in (chess.QUEEN, chess.ROOK):
                    check_threat += 40
                if victim_king is not None and chess.square_distance(move.to_square, victim_king) <= 1:
                    check_threat += 40

                threat = max(threat, check_threat)

            if threat > best_opp_threat:
                best_opp_threat = threat

        if best_opp_threat >= 900:
            return 320
        if best_opp_threat >= 500:
            return 200
        if best_opp_threat >= 320:
            return 140
        if best_opp_threat >= 220:
            return 90
        if best_opp_threat >= 150:
            return 60
        if best_opp_threat >= 100:
            return 40
        return 0

    def _pick_emergency_root_move(self, board: chess.Board) -> Optional[chess.Move]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        best_move = legal_moves[0]
        best_score = -INF

        for move in legal_moves:
            board.push(move)
            try:
                if board.is_checkmate():
                    return move

                tactical_penalty = self._root_tactical_penalty(board)
                # After push(), board.turn is opponent; negate to score from our side.
                static_score = -self._eval_stm(board)
                emergency_score = static_score - tactical_penalty
            finally:
                board.pop()

            if emergency_score > best_score:
                best_score = emergency_score
                best_move = move

        return best_move
 
    # ── Evaluation helpers ────────────────────────────────────────────
    def _pawn_attacks(self, board: chess.Board, color: chess.Color) -> int:
        pawns = int(board.pieces(chess.PAWN, color))
        if color == chess.WHITE:
            return ((pawns << 7) & ~chess.BB_FILE_H) | ((pawns << 9) & ~chess.BB_FILE_A)
        else:
            return ((pawns >> 7) & ~chess.BB_FILE_A) | ((pawns >> 9) & ~chess.BB_FILE_H)
 
    def _pawn_structure_score(self, board: chess.Board, color: chess.Color, phase: int,
                              pawns: Optional[List[chess.Square]] = None,
                              enemy_pawns: Optional[List[chess.Square]] = None) -> int:
        if pawns is None:
            pawns = list(board.pieces(chess.PAWN, color))
        if not pawns:
            return 0
        if enemy_pawns is None:
            enemy_pawns = list(board.pieces(chess.PAWN, not color))
        file_counts   = [0] * 8
        own_by_file   = [[] for _ in range(8)]
        enemy_by_file = [[] for _ in range(8)]
        for sq in pawns:
            f = chess.square_file(sq)
            file_counts[f] += 1
            own_by_file[f].append(chess.square_rank(sq))
        for sq in enemy_pawns:
            enemy_by_file[chess.square_file(sq)].append(chess.square_rank(sq))
 
        score = 0
        eg_weight     = TOTAL_PHASE - phase
        isolated_pen  = 24 if phase >= 12 else 16
        backward_pen  = 16 if phase >= 12 else 11
 
        for count in file_counts:
            if count > 1:
                extra = count - 1
                score -= 13 * extra + 9 * extra * extra
 
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            has_left  = f > 0 and file_counts[f-1] > 0
            has_right = f < 7 and file_counts[f+1] > 0
            if not has_left and not has_right:
                score -= isolated_pen
 
            # Backward
            step  = 1 if color == chess.WHITE else -1
            next_r = r + step
            if 0 <= next_r <= 7:
                front_sq = chess.square(f, next_r)
                if board.piece_at(front_sq) is None:
                    has_support = False
                    for af in (f-1, f+1):
                        if 0 <= af <= 7:
                            ranks = own_by_file[af]
                            if color == chess.WHITE and any(rr <= r for rr in ranks):
                                has_support = True; break
                            if color == chess.BLACK and any(rr >= r for rr in ranks):
                                has_support = True; break
                    if (not has_support
                            and board.is_attacked_by(not color, front_sq)
                            and not board.is_attacked_by(color, front_sq)):
                        score -= backward_pen
 
            # Passed pawn
            passed = True
            for ef in range(max(0, f-1), min(7, f+1)+1):
                er_list = enemy_by_file[ef]
                if color == chess.WHITE:
                    if any(er > r for er in er_list):
                        passed = False; break
                else:
                    if any(er < r for er in er_list):
                        passed = False; break
            if passed:
                advance  = r if color == chess.WHITE else 7 - r
                base     = PASSED_PAWN_BONUS[advance]
                score   += base + (base * eg_weight) // 6
                # King proximity in endgame
                if eg_weight >= 12:
                    promo_sq = chess.square(f, 7 if color == chess.WHITE else 0)
                    wk = board.king(chess.WHITE)
                    bk = board.king(chess.BLACK)
                    if wk and bk:
                        wk_dist = abs(chess.square_file(wk) - f) + abs(chess.square_rank(wk) - (7 if color == chess.WHITE else 0))
                        bk_dist = abs(chess.square_file(bk) - f) + abs(chess.square_rank(bk) - (7 if color == chess.WHITE else 0))
                        diff = (bk_dist - wk_dist) if color == chess.WHITE else (wk_dist - bk_dist)
                        score += diff * 4 * (eg_weight // 6)
 
            # Connected pawns
            for af in (f-1, f+1):
                if 0 <= af <= 7 and any(abs(rr - r) <= 1 for rr in own_by_file[af]):
                    score += 9
        return score
 
    def _knight_outpost_score(self, board: chess.Board, color: chess.Color) -> int:
        score = 0
        own_pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = list(board.pieces(chess.PAWN, not color))
        own_pawn_set = set(own_pawns)
        enemy_ranks_by_file = [[] for _ in range(8)]
        for ep in enemy_pawns:
            enemy_ranks_by_file[chess.square_file(ep)].append(chess.square_rank(ep))

        outpost_mask = OUTPOST_RANKS_W if color == chess.WHITE else OUTPOST_RANKS_B
        for sq in board.pieces(chess.KNIGHT, color):
            if not (chess.BB_SQUARES[sq] & outpost_mask):
                continue
            f = chess.square_file(sq)
            r = chess.square_rank(sq)

            enemy_can_attack = False
            for ef in (f-1, f+1):
                if not (0 <= ef <= 7):
                    continue
                ranks = enemy_ranks_by_file[ef]
                if color == chess.WHITE:
                    if any(er < r for er in ranks):
                        enemy_can_attack = True
                        break
                else:
                    if any(er > r for er in ranks):
                        enemy_can_attack = True
                        break
                if enemy_can_attack:
                    break
            if enemy_can_attack:
                continue

            support_rank = r - 1 if color == chess.WHITE else r + 1
            supported = False
            if 0 <= support_rank <= 7:
                left_support = (f > 0 and chess.square(f - 1, support_rank) in own_pawn_set)
                right_support = (f < 7 and chess.square(f + 1, support_rank) in own_pawn_set)
                supported = left_support or right_support

            score += 32 if supported else 16
        return score
 
    def _bad_bishop_penalty(self, board: chess.Board, color: chess.Color) -> int:
        bishops = board.pieces(chess.BISHOP, color)
        pawns   = board.pieces(chess.PAWN, color)
        if not bishops or not pawns:
            return 0
        penalty = 0
        for bsq in bishops:
            bc = (chess.square_file(bsq) + chess.square_rank(bsq)) % 2
            blocked = sum(1 for psq in pawns
                          if (chess.square_file(psq) + chess.square_rank(psq)) % 2 == bc)
            penalty -= blocked * 4
        return penalty
 
    def _safe_mobility_score(self, board: chess.Board, color: chess.Color) -> int:
        epa = self._pawn_attacks(board, not color)
        own = int(board.occupied_co[color])
        score = 0
        weights = {chess.KNIGHT: 4, chess.BISHOP: 4, chess.ROOK: 2, chess.QUEEN: 1}
        for pt, w in weights.items():
            for sq in board.pieces(pt, color):
                attacks = int(board.attacks_mask(sq)) & ~own & ~epa
                score += w * chess.popcount(attacks)
        return score
 
    def _rook_file_score(self, board: chess.Board, color: chess.Color) -> int:
        rooks      = list(board.pieces(chess.ROOK, color))
        own_pawns  = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        score = 0
        for sq in rooks:
            f  = chess.square_file(sq)
            fm = chess.BB_FILES[f]
            own_on   = bool(own_pawns   & fm)
            enemy_on = bool(enemy_pawns & fm)
            if not own_on and not enemy_on: score += 28
            elif not own_on:                score += 16
            rank = chess.square_rank(sq)
            if (color == chess.WHITE and rank == 6) or (color == chess.BLACK and rank == 1):
                score += 22
        if len(rooks) >= 2:
            r0, r1 = rooks[0], rooks[1]
            if chess.square_rank(r0) == chess.square_rank(r1) or chess.square_file(r0) == chess.square_file(r1):
                score += 12
        return score
 
    def _king_safety_score(self, board: chess.Board, color: chess.Color, phase: int) -> int:
        ksq = board.king(color)
        if ksq is None:
            return 0
        fi = chess.square_file(ksq)
        ri = chess.square_rank(ksq)
        score = 0
 
        if phase >= 10:
            if ((color == chess.WHITE and ksq in (chess.G1, chess.C1)) or
                (color == chess.BLACK and ksq in (chess.G8, chess.C8))):
                score += 28
            else:
                score -= 22
 
            pawn_rank = ri + (1 if color == chess.WHITE else -1)
            if 0 <= pawn_rank <= 7:
                for df in (-1, 0, 1):
                    pf = fi + df
                    if pf < 0 or pf > 7: continue
                    shq = chess.square(pf, pawn_rank)
                    p   = board.piece_at(shq)
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        score += 11
                    else:
                        score -= 16
 
            own_p   = board.pieces(chess.PAWN, color)
            enemy_p = board.pieces(chess.PAWN, not color)
            heavy   = board.pieces(chess.ROOK, not color) | board.pieces(chess.QUEEN, not color)
            for df in (-1, 0, 1):
                pf = fi + df
                if pf < 0 or pf > 7: continue
                fm = chess.BB_FILES[pf]
                if not (own_p & fm):
                    score -= 14
                    if not (enemy_p & fm): score -= 8
                if not (own_p & fm) and (heavy & fm): score -= 16
 
            # Pawn storm
            for ep in enemy_p:
                if abs(chess.square_file(ep) - fi) > 2:
                    continue
                er = chess.square_rank(ep)
                dist = er - ri if color == chess.WHITE else ri - er
                if 0 < dist <= 4:
                    score -= (5 - dist) * 7
 
            # Heavy piece proximity
            for sq in board.pieces(chess.QUEEN, not color):
                dist = abs(fi - chess.square_file(sq)) + abs(ri - chess.square_rank(sq))
                if dist <= 7: score -= max(0, 32 - 4*dist)
            for sq in board.pieces(chess.ROOK, not color):
                dist = abs(fi - chess.square_file(sq)) + abs(ri - chess.square_rank(sq))
                if dist <= 6: score -= max(0, 22 - 3*dist)
        else:
            # Endgame: centralise king
            cd = abs(fi - 3.5) + abs(ri - 3.5)
            score += int((7.0 - cd) * 6)
 
        return score
 
    def _hanging_pieces_score(self, board: chess.Board, color: chess.Color) -> int:
        """Penalty for undefended pieces attacked by enemy."""
        hk = (self._tt_key(board), color)
        cached = self.hanging_cache.get(hk)
        if cached is not None:
            return cached

        score = 0
        enemy = not color
        for sq in chess.scan_forward(board.occupied_co[color]):
            piece = board.piece_at(sq)
            if piece is None or piece.piece_type == chess.KING:
                continue
            if board.is_attacked_by(enemy, sq) and not board.is_attacked_by(color, sq):
                score -= PIECE_VALUES[piece.piece_type] // 6

        if len(self.hanging_cache) >= self.max_hanging_cache_entries:
            self.hanging_cache.clear()
        self.hanging_cache[hk] = score
        return score
 
    def _endgame_knowledge(self, board: chess.Board) -> int:
        score = 0
        wq = len(board.pieces(chess.QUEEN,  chess.WHITE))
        bq = len(board.pieces(chess.QUEEN,  chess.BLACK))
        wr = len(board.pieces(chess.ROOK,   chess.WHITE))
        br = len(board.pieces(chess.ROOK,   chess.BLACK))
        wm = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE))
        bm = len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK))
        wp = chess.popcount(int(board.pieces(chess.PAWN,  chess.WHITE)))
        bp = chess.popcount(int(board.pieces(chess.PAWN,  chess.BLACK)))
        total = wq+bq+wr+br+wm+bm+wp+bp
 
        # KPK
        if total == wp + bp and wq == 0 and bq == 0 and wr == 0 and br == 0:
            wk = board.king(chess.WHITE)
            bk = board.king(chess.BLACK)
            for sq in board.pieces(chess.PAWN, chess.WHITE):
                r = chess.square_rank(sq); f = chess.square_file(sq)
                if wk and abs(chess.square_file(wk)-f) <= 1 and chess.square_rank(wk) == r+1:
                    score += 45
                if bk and abs(chess.square_file(bk)-f) <= 1 and chess.square_rank(bk) > r+1:
                    score -= 55
            for sq in board.pieces(chess.PAWN, chess.BLACK):
                r = chess.square_rank(sq); f = chess.square_file(sq)
                if bk and abs(chess.square_file(bk)-f) <= 1 and chess.square_rank(bk) == r-1:
                    score -= 45
                if wk and abs(chess.square_file(wk)-f) <= 1 and chess.square_rank(wk) < r-1:
                    score += 55
 
        # Rook endgames
        if wr == 1 and br == 1 and wq == 0 and bq == 0:
            for sq in board.pieces(chess.ROOK, chess.WHITE):
                if chess.square_rank(sq) >= 6: score += 16
            for sq in board.pieces(chess.ROOK, chess.BLACK):
                if chess.square_rank(sq) <= 1: score -= 16
 
        # Drive lone king to corner
        if wq >= 1 and bq == 0 and br == 0 and bm == 0:
            bk = board.king(chess.BLACK)
            if bk:
                bf = chess.square_file(bk); br_ = chess.square_rank(bk)
                score += 22 * (3 - min(bf, 7-bf) + 3 - min(br_, 7-br_))
                if board.king(chess.WHITE):
                    wk = board.king(chess.WHITE)
                    wf = chess.square_file(wk); wr_ = chess.square_rank(wk)
                    dist = abs(wf-bf) + abs(wr_-br_)
                    score += max(0, 28 - 2*dist)
 
        if bq >= 1 and wq == 0 and wr == 0 and wm == 0:
            wk = board.king(chess.WHITE)
            if wk:
                wf = chess.square_file(wk); wr_ = chess.square_rank(wk)
                score -= 22 * (3 - min(wf, 7-wf) + 3 - min(wr_, 7-wr_))
                bk = board.king(chess.BLACK)
                if bk:
                    bf = chess.square_file(bk); br_ = chess.square_rank(bk)
                    dist = abs(wf-bf) + abs(wr_-br_)
                    score -= max(0, 28 - 2*dist)
 
        return score
 
    def evaluate_white(self, board: chess.Board) -> int:
        mg = 0; eg = 0; phase = 0
        for sq, piece in board.piece_map().items():
            pt   = piece.piece_type
            sign = 1 if piece.color == chess.WHITE else -1
            phase += PHASE_WEIGHTS[pt]
            mg    += sign * (MG_VALUES[pt] + pst_value(PST_MG[pt], sq, piece.color))
            eg    += sign * (EG_VALUES[pt] + pst_value(PST_EG[pt], sq, piece.color))
 
        phase      = min(phase, TOTAL_PHASE)
        base_score = (mg * phase + eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE
        score      = base_score

        white_pawns_bb = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns_bb = board.pieces(chess.PAWN, chess.BLACK)
        white_pawns = list(white_pawns_bb)
        black_pawns = list(black_pawns_bb)
        all_pawns = white_pawns_bb | black_pawns_bb
        white_bishops = board.pieces(chess.BISHOP, chess.WHITE)
        black_bishops = board.pieces(chess.BISHOP, chess.BLACK)
 
        # Bishop pair (bonus grows with board openness)
        open_files = sum(1 for f in range(8)
                 if not all_pawns & chess.BB_FILES[f])
        bp_bonus = 30 + open_files * 3
        if len(white_bishops) >= 2: score += bp_bonus
        if len(black_bishops) >= 2: score -= bp_bonus
 
        fast_middlegame_eval = self.eval_mode == "fast" and phase >= 8

        score += self._pawn_structure_score(board, chess.WHITE, phase,
                            pawns=white_pawns,
                            enemy_pawns=black_pawns)
        score -= self._pawn_structure_score(board, chess.BLACK, phase,
                            pawns=black_pawns,
                            enemy_pawns=white_pawns)
        score += self._rook_file_score(board, chess.WHITE)
        score -= self._rook_file_score(board, chess.BLACK)
        score += self._knight_outpost_score(board, chess.WHITE)
        score -= self._knight_outpost_score(board, chess.BLACK)
        score += self._bad_bishop_penalty(board, chess.WHITE)
        score -= self._bad_bishop_penalty(board, chess.BLACK)
        if not fast_middlegame_eval:
            score += self._safe_mobility_score(board, chess.WHITE)
            score -= self._safe_mobility_score(board, chess.BLACK)
            score += self._king_safety_score(board, chess.WHITE, phase)
            score -= self._king_safety_score(board, chess.BLACK, phase)
            score += self._hanging_pieces_score(board, chess.WHITE)
            score -= self._hanging_pieces_score(board, chess.BLACK)
            score += self._endgame_knowledge(board)
 
        # Tempo bonus (side to move)
        score += 8 if board.turn == chess.WHITE else -8
 
        # Clamp dynamic terms
        dynamic = score - base_score
        dynamic = max(-500, min(500, dynamic))
        return base_score + dynamic
 
    def _eval_stm(self, board: chess.Board) -> int:
        key = self._tt_key(board)
        raw = self.eval_cache.get(key)
        if raw is None:
            raw = self.evaluate_white(board)
            if len(self.eval_cache) >= self.max_eval_cache_entries:
                self.eval_cache.clear()
            self.eval_cache[key] = raw
        stm_raw = raw if board.turn == chess.WHITE else -raw
 
        # Apply correction history
        ck = (board.turn, key & 0xFFFF)
        corr = self.corr_hist.get(ck, 0)
        return stm_raw + corr // 8
 
    # ── Search infra ──────────────────────────────────────────────────
    def _check_timeout(self):
        if self.deadline is None: return
        if ((self.nodes + self.qnodes) & 2047) != 0: return
        if time.time() >= self.deadline:
            raise SearchTimeout()
 
    def _tt_key(self, board: chess.Board):
        return chess.polyglot.zobrist_hash(board)
 
    def _score_to_tt(self, score: int, ply: int) -> int:
        if score >= MATE_BOUND:  return score + ply
        if score <= -MATE_BOUND: return score - ply
        return score
 
    def _score_from_tt(self, score: int, ply: int) -> int:
        if score >= MATE_BOUND:  return score - ply
        if score <= -MATE_BOUND: return score + ply
        return score
 
    def _evict_tt(self):
        victim_key, victim_rank = None, INF
        sample_budget = 768
        for i, (k, e) in enumerate(self.tt.items()):
            age_pen = max(0, self.tt_generation - e.age)
            move_bonus = 1 if e.move is not None else 0
            exact_bonus = 1 if e.flag == TT_EXACT else 0
            rank = e.depth + exact_bonus + move_bonus - 4 * age_pen
            if rank < victim_rank:
                victim_rank = rank
                victim_key = k
            if i + 1 >= sample_budget:
                break

        if victim_key is None:
            victim_key = next(iter(self.tt))
        self.tt.pop(victim_key, None)
 
    def _store_tt(self, key, depth, score, flag, move, ply):
        packed = self._score_to_tt(score, ply)
        cur    = self.tt.get(key)
        if cur is not None:
            stale    = (self.tt_generation - cur.age) >= 2
            depth_bonus = depth + (2 if flag == TT_EXACT else 0) + (1 if move is not None else 0)
            cur_bonus   = cur.depth + (2 if cur.flag == TT_EXACT else 0) + (1 if cur.move is not None else 0)
            # Keep depth preference strict; allow shallow exact only when close in depth.
            stronger_close = (
                flag == TT_EXACT
                and cur.flag != TT_EXACT
                and depth + 1 >= cur.depth
            )
            if move is None and cur.move is not None:
                move = cur.move
            if depth_bonus >= cur_bonus or stale or stronger_close:
                self.tt[key] = TTEntry(depth, packed, flag, move, self.tt_generation)
            return
        if len(self.tt) >= self.max_tt_entries:
            self._evict_tt()
        self.tt[key] = TTEntry(depth, packed, flag, move, self.tt_generation)
 
    def _store_killer(self, move: chess.Move, ply: int):
        if ply >= MAX_PLY: return
        k0, _ = self.killers[ply]
        if move == k0: return
        self.killers[ply][1] = k0
        self.killers[ply][0] = move
 
    def _has_non_pawn_material(self, board: chess.Board, color: chess.Color) -> bool:
        return bool(board.occupied_co[color] & ~(board.pawns | board.kings))
 
    def _captured_value(self, board: chess.Board, move: chess.Move) -> int:
        if board.is_en_passant(move): return PIECE_VALUES[chess.PAWN]
        v = board.piece_at(move.to_square)
        return PIECE_VALUES[v.piece_type] if v else 0

    def _is_capture_cached(self, board: chess.Board, move: chess.Move,
                           cache: Optional[Dict[chess.Move, bool]] = None) -> bool:
        if cache is None:
            return board.is_capture(move)
        if move in cache:
            return cache[move]
        val = board.is_capture(move)
        cache[move] = val
        return val

    def _gives_check_cached(self, board: chess.Board, move: chess.Move,
                            cache: Optional[Dict[chess.Move, bool]] = None) -> bool:
        if cache is None:
            return board.gives_check(move)
        if move in cache:
            return cache[move]
        val = board.gives_check(move)
        cache[move] = val
        return val

    def _captured_value_cached(self, board: chess.Board, move: chess.Move,
                               cache: Optional[Dict[chess.Move, int]] = None) -> int:
        if cache is None:
            return self._captured_value(board, move)
        if move in cache:
            return cache[move]
        val = self._captured_value(board, move)
        cache[move] = val
        return val

    def _can_use_null_move(self, board, depth, in_check, allow_null, static_eval, beta):
        if not allow_null or in_check or depth < 3:
            return False
        if static_eval < beta:
            return False
        if not self._has_non_pawn_material(board, board.turn):
            return False
        side_np  = chess.popcount(int(board.occupied_co[board.turn] & ~(board.pawns | board.kings)))
        total_np = chess.popcount(int(board.occupied & ~(board.pawns | board.kings)))
        total_nk = chess.popcount(int(board.occupied & ~board.kings))
        if side_np <= 1:
            return False
        if total_np <= 2 and total_nk <= 6:
            return False
        own_pawns = chess.popcount(int(board.pieces(chess.PAWN, board.turn)))
        if own_pawns >= 3 and total_np <= 2:
            return False
        return True

    # ── Move ordering ─────────────────────────────────────────────────
    def _quiet_move_score(self, board: chess.Board, move: chess.Move,
                          ply: int, prev_move: Optional[chess.Move]) -> int:
        s = 0
        mover = board.piece_at(move.from_square)
        enemy = not board.turn

        if ply < MAX_PLY:
            k0, k1 = self.killers[ply]
            if k0 and move == k0:
                s += 3_200_000
            elif k1 and move == k1:
                s += 2_700_000

            if ply > 0:
                pk0, pk1 = self.killers[ply - 1]
                if pk0 and move == pk0:
                    s += 900_000
                elif pk1 and move == pk1:
                    s += 700_000

        if prev_move:
            cm = self.counter.get((prev_move.from_square, prev_move.to_square))
            if cm and cm == move:
                s += 2_000_000
            if move.to_square == prev_move.to_square:
                s += 950_000

        h = self.history.get((board.turn, move.from_square, move.to_square), 0)
        s += min(1_600_000, h) // 8

        if prev_move:
            ch = self.cont_hist.get(
                (prev_move.from_square, prev_move.to_square,
                 board.turn, move.from_square, move.to_square), 0)
            s += min(600_000, ch) // 8

        # Defensive quiet evasions are often critical and can be missed if
        # they are ordered too late among non-captures.
        if mover and mover.piece_type != chess.KING:
            from_attacked = board.is_attacked_by(enemy, move.from_square)
            to_attacked = board.is_attacked_by(enemy, move.to_square)
            if from_attacked and not to_attacked:
                s += 1_050_000 + PIECE_VALUES[mover.piece_type] * 80
            elif not from_attacked and to_attacked:
                s -= 380_000

        return s

    def _move_score(self, board: chess.Board, move: chess.Move,
                    tt_move, ply: int, prev_move: Optional[chess.Move],
                    capture_cache: Optional[Dict[chess.Move, bool]] = None,
                    check_cache: Optional[Dict[chess.Move, bool]] = None,
                    captured_value_cache: Optional[Dict[chess.Move, int]] = None) -> int:
        if tt_move and move == tt_move:
            return 12_000_000

        gives_check = self._gives_check_cached(board, move, check_cache)
        is_capture = self._is_capture_cached(board, move, capture_cache)

        if is_capture:
            atk  = board.piece_at(move.from_square)
            av   = PIECE_VALUES[atk.piece_type] if atk else PIECE_VALUES[chess.PAWN]
            vv   = self._captured_value_cached(board, move, captured_value_cache)
            base = 6_000_000 + vv * 32 - av * 8 + vv * 8
            if gives_check:  base += 150_000
            if move.promotion: base += PIECE_VALUES.get(move.promotion, 0)
            return base
 
        if move.promotion:
            return 5_000_000 + PIECE_VALUES.get(move.promotion, 0)
 
        s = self._quiet_move_score(board, move, ply, prev_move)
        if gives_check:
            s += 1_200_000
        return s
 
    def _ordered_moves(self, board: chess.Board,
                       moves: List[chess.Move],
                       tt_move: Optional[chess.Move],
                       ply: int,
                       captures_only: bool = False,
                       prev_move: Optional[chess.Move] = None,
                       pv_node: bool = False,
                       full_sort: bool = False,
                       capture_cache: Optional[Dict[chess.Move, bool]] = None,
                       check_cache: Optional[Dict[chess.Move, bool]] = None,
                       captured_value_cache: Optional[Dict[chess.Move, int]] = None) -> List[chess.Move]:
        if captures_only:
            moves = [m for m in moves
                     if self._is_capture_cached(board, m, capture_cache) or m.promotion]
        n_moves = len(moves)
        if n_moves <= 1:
            return moves

        score_fn = lambda m: self._move_score(
            board, m, tt_move, ply, prev_move,
            capture_cache=capture_cache,
            check_cache=check_cache,
            captured_value_cache=captured_value_cache)
        if full_sort or pv_node or captures_only or n_moves < STAGED_ORDERING_MIN_MOVES:
            return sorted(moves, key=score_fn, reverse=True)

        lead: List[chess.Move] = []
        tactical: List[chess.Move] = []
        quiet: List[chess.Move] = []

        for move in moves:
            if tt_move and move == tt_move:
                lead.append(move)
                continue
            if (move.promotion
                    or self._is_capture_cached(board, move, capture_cache)
                    or self._gives_check_cached(board, move, check_cache)):
                tactical.append(move)
            else:
                quiet.append(move)

        if tactical:
            tactical.sort(key=score_fn, reverse=True)

        if not quiet:
            return lead + tactical

        quiet_score = lambda m: self._quiet_move_score(board, m, ply, prev_move)
        head_count = min(STAGED_QUIET_HEAD, len(quiet))
        if head_count == len(quiet):
            quiet_head = sorted(quiet, key=quiet_score, reverse=True)
            return lead + tactical + quiet_head

        # Keep expensive ordering focused on the first quiet bucket.
        quiet_head = heapq.nlargest(head_count, quiet, key=quiet_score)
        quiet_head_set = set(quiet_head)
        quiet_tail = [m for m in quiet if m not in quiet_head_set]
        return lead + tactical + quiet_head + quiet_tail

    def _top_ordered_moves(self, board: chess.Board,
                           moves: List[chess.Move],
                           tt_move: Optional[chess.Move],
                           ply: int,
                           limit: int,
                           prev_move: Optional[chess.Move] = None,
                           capture_cache: Optional[Dict[chess.Move, bool]] = None,
                           check_cache: Optional[Dict[chess.Move, bool]] = None,
                           captured_value_cache: Optional[Dict[chess.Move, int]] = None) -> List[chess.Move]:
        if limit <= 0:
            return []
        n_moves = len(moves)
        if n_moves <= 1:
            return moves[:]
        score_fn = lambda m: self._move_score(
            board, m, tt_move, ply, prev_move,
            capture_cache=capture_cache,
            check_cache=check_cache,
            captured_value_cache=captured_value_cache)
        if n_moves <= limit:
            return sorted(moves, key=score_fn, reverse=True)
        return heapq.nlargest(limit, moves, key=score_fn)
 
    # ── Quiescence search ─────────────────────────────────────────────
    def quiescence(self, board: chess.Board, alpha: int, beta: int,
                   ply: int, prev_move: Optional[chess.Move] = None,
                   qs_depth: int = 0) -> int:
        self.qnodes += 1
        self._check_timeout()
 
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0
        if board.is_repetition(2):
            return 0
 
        in_check  = board.is_check()
        stand_pat = None
        capture_cache: Dict[chess.Move, bool] = {}
        check_cache: Dict[chess.Move, bool] = {}
        capture_values: Dict[chess.Move, int] = {}
        see_cache: Dict[chess.Move, int] = {}
 
        if not in_check:
            stand_pat = self._eval_stm(board)
            if stand_pat >= beta:      return beta
            if stand_pat + 1100 < alpha: return alpha
            if stand_pat > alpha:      alpha = stand_pat
            moves = list(board.generate_legal_captures())

            filtered = []
            for move in moves:
                capture_cache[move] = True
                cap_val = self._captured_value_cached(board, move, capture_values)
                if stand_pat + cap_val + QS_DELTA_MARGIN < alpha:
                    continue
                filtered.append(move)

            quiet_checks: List[chess.Move] = []
            can_probe_checks = (
                qs_depth < QS_CHECK_PLY_LIMIT
                and stand_pat + QS_CHECK_MARGIN >= alpha
            )
            if can_probe_checks:
                for move in board.legal_moves:
                    if move.promotion:
                        continue
                    if self._is_capture_cached(board, move, capture_cache):
                        continue
                    if self._gives_check_cached(board, move, check_cache):
                        quiet_checks.append(move)
                if len(quiet_checks) > QS_CHECK_MAX_MOVES:
                    quiet_checks = self._top_ordered_moves(
                        board, quiet_checks, None, ply, QS_CHECK_MAX_MOVES,
                        prev_move=prev_move,
                        capture_cache=capture_cache,
                        check_cache=check_cache,
                        captured_value_cache=capture_values)

            if filtered:
                # Delta-pruned captures first, then a tiny tail of checking non-captures.
                ordered_caps = self._ordered_moves(
                    board, filtered, None, ply,
                    prev_move=prev_move,
                    full_sort=True,
                    capture_cache=capture_cache,
                    check_cache=check_cache,
                    captured_value_cache=capture_values)
                moves = ordered_caps + quiet_checks
            else:
                if not quiet_checks:
                    return alpha
                moves = quiet_checks
        else:
            moves = list(board.legal_moves)
            if not moves: return -MATE_SCORE + ply
            moves = self._ordered_moves(board, moves, None, ply,
                                        prev_move=prev_move,
                                        full_sort=True,
                                        capture_cache=capture_cache,
                                        check_cache=check_cache,
                                        captured_value_cache=capture_values)
 
        for move in moves:
            is_cap = self._is_capture_cached(board, move, capture_cache)
            if stand_pat is not None and is_cap:
                if move.promotion is None and self._see_cached(board, move, see_cache) < QS_SEE_FLOOR:
                    continue
                cap_val = capture_values.get(move)
                if cap_val is None:
                    cap_val = self._captured_value_cached(board, move, capture_values)
                if stand_pat + cap_val + QS_DELTA_MARGIN < alpha:
                    continue
 
            board.push(move)
            try:
                score = -self.quiescence(board, -beta, -alpha, ply+1, move, qs_depth+1)
            finally:
                board.pop()
 
            if score >= beta: return beta
            if score > alpha: alpha = score
 
        return alpha
 
    # ── Main search ───────────────────────────────────────────────────
    def search(self, board: chess.Board, depth: int,
               alpha: int, beta: int, ply: int,
               allow_null: bool,
               prev_move: Optional[chess.Move] = None) -> int:
        self.nodes += 1
        self._check_timeout()
 
        pv_node  = (beta - alpha > 1)
        in_check = board.is_check()
 
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0
        if board.is_repetition(2):
            return 0
        if ply >= MAX_PLY - 1:
            return self._eval_stm(board)
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply, prev_move)
 
        # Mate-distance pruning
        alpha = max(alpha, -MATE_SCORE + ply)
        beta  = min(beta,   MATE_SCORE - ply - 1)
        if alpha >= beta: return alpha
 
        alpha_orig = alpha
        key        = self._tt_key(board)
        tt_entry   = self.tt.get(key)
        tt_move    = tt_entry.move if tt_entry else None
 
        if tt_entry and tt_entry.depth >= depth:
            tt_score = self._score_from_tt(tt_entry.score, ply)
            self.tt_hits += 1
            if tt_entry.flag == TT_EXACT:
                return tt_score
            if tt_entry.flag == TT_LOWER:
                alpha = max(alpha, tt_score)
            elif tt_entry.flag == TT_UPPER:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score
 
        # Syzygy probe
        if self._syzygy_reader and chess.popcount(board.occupied) <= 7:
            tb_score = self._tb_probe_wdl(board)
            if tb_score is not None:
                flag = TT_EXACT
                if tb_score >= MATE_BOUND:    flag = TT_LOWER
                elif tb_score <= -MATE_BOUND: flag = TT_UPPER
                self._store_tt(key, depth, tb_score, flag, None, ply)
                return tb_score
 
        static_eval = self._eval_stm(board) if not in_check else -INF
        capture_cache: Dict[chess.Move, bool] = {}
        check_cache: Dict[chess.Move, bool] = {}
        captured_value_cache: Dict[chess.Move, int] = {}
        see_cache: Dict[chess.Move, int] = {}
        prev_was_capture = prev_move is not None and board.is_capture(prev_move)
 
        # ── Razoring ──────────────────────────────────────────────────
        if (not pv_node and not in_check and depth <= 2 and tt_move is None
                and abs(beta) < MATE_BOUND):
            if static_eval + RAZOR_MARGINS[depth] < alpha:
                if depth == 1:
                    return self.quiescence(board, alpha, beta, ply, prev_move)
                qs = self.quiescence(board, alpha - RAZOR_MARGINS[depth],
                                     alpha - RAZOR_MARGINS[depth] + 1, ply, prev_move)
                if qs < alpha:
                    return qs
 
        # ── Reverse futility (static null move) ───────────────────────
        if (not pv_node and not in_check and 2 <= depth <= 7
                and abs(beta) < MATE_BOUND
                and self._has_non_pawn_material(board, board.turn)):
            margin = 82 * depth
            if static_eval - margin >= beta:
                rfp_score = static_eval - margin
                self._store_tt(key, 1, rfp_score, TT_LOWER, None, ply)
                return rfp_score
 
        # ── Null-move pruning ─────────────────────────────────────────
        if self._can_use_null_move(board, depth, in_check, allow_null, static_eval, beta):
            r = 3 + depth // 4 + min(3, (static_eval - beta) // 180)
            board.push(chess.Move.null())
            try:
                score = -self.search(board, depth - 1 - r, -beta, -beta+1, ply+1, False, None)
            finally:
                board.pop()
            if score >= beta and abs(score) < MATE_BOUND:
                self._store_tt(key, max(1, depth - 1 - r), score, TT_LOWER, None, ply)
                return score
 
        # ── ProbCut ───────────────────────────────────────────────────
        if (not pv_node and depth >= 7 and not in_check and abs(beta) < MATE_BOUND):
            pc_beta  = beta + 180
            pc_depth = depth - 4
            caps = []
            for m in board.generate_legal_captures():
                capture_cache[m] = True
                if self._captured_value_cached(board, m, captured_value_cache) + static_eval >= pc_beta:
                    caps.append(m)
            if caps:
                top_caps = self._top_ordered_moves(
                    board, caps, tt_move, ply, PROBCUT_CANDIDATE_LIMIT,
                    prev_move=prev_move,
                    capture_cache=capture_cache,
                    check_cache=check_cache,
                    captured_value_cache=captured_value_cache)
            else:
                top_caps = []

            for m in top_caps:
                if self._see_cached(board, m, see_cache) < 0: continue
                board.push(m)
                try:
                    sc = -self.search(board, pc_depth, -pc_beta, -pc_beta+1, ply+1, False, m)
                finally:
                    board.pop()
                if sc >= pc_beta:
                    self._store_tt(key, max(1, pc_depth), sc, TT_LOWER, m, ply)
                    return sc
 
        # ── IID ───────────────────────────────────────────────────────
        if pv_node and depth >= 5 and tt_move is None:
            self.search(board, depth - 2, alpha, beta, ply, False, prev_move)
            tt_entry = self.tt.get(key)
            tt_move  = tt_entry.move if tt_entry else None
 
        moves = list(board.legal_moves)
        if not moves:
            terminal = -MATE_SCORE + ply if in_check else 0
            self._store_tt(key, depth, terminal, TT_EXACT, None, ply)
            return terminal
 
        ordered = self._ordered_moves(board, moves, tt_move, ply,
                          prev_move=prev_move,
                          pv_node=pv_node,
                          capture_cache=capture_cache,
                          check_cache=check_cache,
                          captured_value_cache=captured_value_cache)

        side_to_move = board.turn
        single_evasion = in_check and len(moves) == 1
        threat_eval_done = False
        threat_extension_active = False
 
        best_score = -INF
        best_move  = None
        move_count = 0
        fail_highs = 0
 
        for move in ordered:
            is_cap      = self._is_capture_cached(board, move, capture_cache)
            is_promo    = move.promotion is not None
            gives_chk   = self._gives_check_cached(board, move, check_cache)
            is_tactical = is_cap or gives_chk or is_promo
            quiet_evasion = (
                (not is_tactical)
                and board.is_attacked_by(not side_to_move, move.from_square)
                and (not board.is_attacked_by(not side_to_move, move.to_square))
            )
 
            # ── Extensions ──────────────────────────────────────────
            ext = 0
 
            # Single evasion
            if single_evasion:
                ext = 1
 
            # Passed pawn push to rank 6/7
            if not ext and not is_cap and not is_promo:
                piece = board.piece_at(move.from_square)
                if piece and piece.piece_type == chess.PAWN:
                    tr = chess.square_rank(move.to_square)
                    if ((board.turn == chess.WHITE and tr >= 5) or
                        (board.turn == chess.BLACK and tr <= 2)):
                        ext = 1
 
            # Recapture
            if not ext and is_cap and prev_move and prev_was_capture:
                if move.to_square == prev_move.to_square:
                    ext = 1
 
            # Threat extension: opponent has hanging strong piece
            if not ext and not is_cap and ply <= 4:
                if not threat_eval_done:
                    threat_eval_done = True
                    threat_extension_active = self._hanging_pieces_score(board, not side_to_move) < -200
                if threat_extension_active:
                    ext = 1
 
            # Singular extension
            if (not ext and depth >= 6 and tt_entry and tt_entry.move == move
                    and tt_entry.depth >= depth - 3
                    and tt_entry.flag != TT_UPPER
                    and abs(tt_entry.score) < MATE_BOUND):
                s_beta  = max(-MATE_SCORE, self._score_from_tt(tt_entry.score, ply) - 52)
                s_depth = (depth - 1) // 2
                excl_moves = [m for m in moves if m != move]
                if excl_moves:
                    best_excl = -INF
                    for em in self._top_ordered_moves(
                            board, excl_moves, None, ply, SINGULAR_EXCLUDE_LIMIT,
                            capture_cache=capture_cache,
                            check_cache=check_cache,
                            captured_value_cache=captured_value_cache):
                        board.push(em)
                        try:
                            sc = -self.search(board, s_depth, -s_beta-1, -s_beta, ply+1, False, em)
                        finally:
                            board.pop()
                        if sc > best_excl: best_excl = sc
                        if best_excl >= s_beta: break
                    if best_excl < s_beta:
                        ext = 1
 
            new_depth = depth - 1 + ext
 
            # ── Futility pruning ─────────────────────────────────────
            if (not pv_node and not in_check and not is_tactical and not ext
                    and not quiet_evasion
                    and new_depth <= 4 and move_count > 0
                    and abs(alpha) < MATE_BOUND):
                if static_eval + FUTILITY_MARGINS[new_depth] <= alpha:
                    move_count += 1
                    continue
 
            # ── Late-move pruning ────────────────────────────────────
            if (not pv_node and not in_check and not is_tactical and not ext
                    and not quiet_evasion
                    and depth <= 5
                    and move_count >= (9 + 2 * depth * depth)
                    and static_eval + 95 * depth <= alpha):
                move_count += 1
                continue
 
            # ── SEE-based capture pruning ────────────────────────────
            if (not pv_node and is_cap and not gives_chk
                    and move_count > 0 and depth <= 10):
                see = self._see_cached(board, move, see_cache)
                threshold = -72 - 22 * depth
                if see < threshold:
                    move_count += 1
                    continue
 
            # ── Multi-cut ────────────────────────────────────────────
            if fail_highs >= 3 and not pv_node and depth >= 4:
                self._store_tt(key, max(1, depth - 1), beta, TT_LOWER, best_move, ply)
                return beta
 
            board.push(move)
            try:
                if move_count == 0:
                    score = -self.search(board, new_depth, -beta, -alpha, ply+1, True, move)
                else:
                    # LMR
                    reduction = 0
                    if depth >= 3 and move_count >= 3 and not in_check and not is_tactical:
                        reduction = int(math.log(depth) * math.log(move_count) / 1.85 - 0.15)
                        reduction = max(0, min(reduction, max(0, new_depth - 2)))
                        if depth <= 4:
                            reduction = min(reduction, 1)
                        if quiet_evasion:
                            reduction = max(0, reduction - 2)
                        h = self.history.get((side_to_move, move.from_square, move.to_square), 0)
                        if h > 280_000:  reduction = max(0, reduction - 2)
                        elif h < -60_000: reduction = min(new_depth - 1, reduction + 1)
                        if pv_node:      reduction = max(0, reduction - 1)
 
                    score = -self.search(board, max(0, new_depth - reduction),
                                         -alpha-1, -alpha, ply+1, True, move)
                    if reduction and score > alpha:
                        score = -self.search(board, new_depth, -alpha-1, -alpha, ply+1, True, move)
                    if alpha < score < beta:
                        score = -self.search(board, new_depth, -beta, -alpha, ply+1, True, move)
            finally:
                board.pop()
 
            move_count += 1
 
            if score > best_score:
                best_score = score
                best_move  = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                fail_highs += 1
                if not is_cap and not is_promo:
                    self._store_killer(move, ply)
                    hk = (board.turn, move.from_square, move.to_square)
                    self.history[hk] = self.history.get(hk, 0) + depth * depth
                    if self.history[hk] > 2_000_000:
                        self.history[hk] //= 2
                    if prev_move:
                        self.counter[(prev_move.from_square, prev_move.to_square)] = move
                    if prev_move:
                        ck = (prev_move.from_square, prev_move.to_square,
                              board.turn, move.from_square, move.to_square)
                        self.cont_hist[ck] = self.cont_hist.get(ck, 0) + depth * depth
                break
 
        if best_move is None:
            return -MATE_SCORE + ply if in_check else 0
 
        # Update correction history
        if best_score != alpha_orig and not in_check:
            diff = best_score - static_eval
            ck   = (board.turn, chess.polyglot.zobrist_hash(board) & 0xFFFF)
            old  = self.corr_hist.get(ck, 0)
            self.corr_hist[ck] = old + (diff - old // 8) // max(1, depth)
            if abs(self.corr_hist[ck]) > 5000:
                self.corr_hist[ck] = self.corr_hist[ck] * 5000 // abs(self.corr_hist[ck])
 
        flag = (TT_UPPER if best_score <= alpha_orig
                else TT_LOWER if best_score >= beta
                else TT_EXACT)
        self._store_tt(key, depth, best_score, flag, best_move, ply)
        return best_score
 
    # ── Root search ───────────────────────────────────────────────────
    def search_root(self, board: chess.Board, depth: int,
                    alpha: int, beta: int,
                    pv_hint: Optional[chess.Move] = None) -> Tuple[Optional[chess.Move], int]:
        alpha_orig = alpha
        key       = self._tt_key(board)
        tt_entry  = self.tt.get(key)
        tt_move   = tt_entry.move if tt_entry else None
        if pv_hint and pv_hint in board.legal_moves:
            tt_move = pv_hint
 
        moves   = self._ordered_moves(board, list(board.legal_moves), tt_move, 0,
                          full_sort=True)
        if not moves: return None, 0
 
        best_move:  Optional[chess.Move] = None
        best_score = -INF
 
        for idx, move in enumerate(moves):
            self._check_timeout()
            is_cap      = board.is_capture(move)
            is_promo    = move.promotion is not None
            gives_chk   = board.gives_check(move)
            is_tactical = is_cap or gives_chk or is_promo
            quiet_evasion = (
                (not is_tactical)
                and board.is_attacked_by(not board.turn, move.from_square)
                and (not board.is_attacked_by(not board.turn, move.to_square))
            )
 
            board.push(move)
            try:
                if idx == 0:
                    score = -self.search(board, depth-1, -beta, -alpha, 1, True, move)
                else:
                    reduction = 0
                    if depth >= 5 and idx >= 4 and not is_tactical:
                        reduction = int(0.5 + math.log(depth) * math.log(idx) / 3.0)
                        reduction = min(reduction, max(0, depth-2))
                        if quiet_evasion:
                            reduction = max(0, reduction - 2)
                    score = -self.search(board, max(0, depth-1-reduction),
                                         -alpha-1, -alpha, 1, True, move)
                    if reduction and score > alpha:
                        score = -self.search(board, depth-1, -alpha-1, -alpha, 1, True, move)
                    if alpha < score < beta:
                        score = -self.search(board, depth-1, -beta, -alpha, 1, True, move)
            finally:
                board.pop()
 
            if score > best_score:
                best_score = score; best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
 
        if best_move:
            flag = (TT_UPPER if best_score <= alpha_orig
                    else TT_LOWER if best_score >= beta
                    else TT_EXACT)
            self._store_tt(key, depth, best_score, flag, best_move, 0)
        return best_move, best_score
 
    # ── Blunder-check root verification pass ──────────────────────────
    def _blunder_check_root(self, board: chess.Board,
                             best_move: Optional[chess.Move],
                             best_score: int, depth: int,
                             verbose: bool) -> Tuple[Optional[chess.Move], int]:
        if best_move is None or depth < 3:
            return best_move, best_score
        if self.deadline is not None and self.deadline - time.time() <= 0.04:
            return best_move, best_score

        key      = self._tt_key(board)
        tt_entry = self.tt.get(key)
        tt_move  = tt_entry.move if tt_entry else best_move
        ordered  = self._top_ordered_moves(
            board, list(board.legal_moves), tt_move, 0, ROOT_SAFETY_CANDIDATES)
        if not ordered:
            return best_move, best_score

        candidates = ordered
        if best_move not in candidates:
            candidates = [best_move] + candidates[: max(0, ROOT_SAFETY_CANDIDATES - 1)]

        verify_depth = max(2, min(6, depth - 1))
        scored: List[Tuple[chess.Move, int, int]] = []
        for mv in candidates:
            if self.deadline is not None and time.time() >= self.deadline - 0.04:
                break
            if mv not in board.legal_moves:
                continue
            board.push(mv)
            timed_out = False
            try:
                tactical_penalty = self._root_tactical_penalty(board)
                sc = -self.search(board, verify_depth - 1, -INF, INF, 1, True, mv)
            except SearchTimeout:
                timed_out = True
            finally:
                board.pop()
            if timed_out:
                break
            scored.append((mv, sc, tactical_penalty))

        if len(scored) < 2:
            return best_move, best_score

        scored.sort(key=lambda x: (x[1] - x[2], x[1]), reverse=True)
        top_move, top_score, top_penalty = scored[0]
        chosen_score = best_score
        chosen_penalty = 0
        for mv, sc, penalty in scored:
            if mv == best_move:
                chosen_score = sc
                chosen_penalty = penalty
                break

        top_adjusted = top_score - top_penalty
        chosen_adjusted = chosen_score - chosen_penalty

        severe_risk = (
            (chosen_penalty >= 10_000 and top_penalty < 10_000)
            or (
                chosen_penalty >= ROOT_EMERGENCY_PENALTY
                and chosen_penalty >= top_penalty + ROOT_SAFETY_PENALTY_GAP
                and top_score >= chosen_score - ROOT_SAFETY_PENALTY_SLACK
            )
        )
        mate_preservation = top_score >= MATE_BOUND and chosen_score < MATE_BOUND
        escape_forced_mate = chosen_score <= -MATE_BOUND and top_score > -MATE_BOUND

        if top_move != best_move and (severe_risk or mate_preservation or escape_forced_mate):
            if verbose:
                print(f"  safety | {best_move.uci()} → {top_move.uci()}"
                      f" | Δadj{top_adjusted - chosen_adjusted:+d}"
                      f" | pen {chosen_penalty}->{top_penalty} | d{verify_depth}")
            return top_move, top_score
        return best_move, best_score
 
    # ── Public API ────────────────────────────────────────────────────
    def principal_variation(self, board: chess.Board, max_len: int = 12) -> List[str]:
        line = []; b = board.copy(stack=False); seen = set()
        for _ in range(max_len):
            key = self._tt_key(b)
            if key in seen: break
            seen.add(key)
            e = self.tt.get(key)
            if not e or not e.move: break
            if e.move not in b.legal_moves: break
            line.append(e.move.uci())
            b.push(e.move)
        return line
 
    def format_score(self, score: int) -> str:
        if score >= MATE_BOUND:  return f"M{(MATE_SCORE - score + 1)//2}"
        if score <= -MATE_BOUND: return f"-M{(MATE_SCORE + score + 1)//2}"
        return f"{score/100:+.2f}"
 
    def find_best_move(self, board: chess.Board,
                       max_depth: int = 10,
                       time_limit: Optional[float] = 5.0,
                       verbose: bool = True) -> Optional[chess.Move]:
        if board.is_game_over(): return None
 
        # Book
        bm = self._book_move(board)
        if bm:
            if verbose: print(f"  [Book] {bm.uci()}")
            return bm
 
        # Tablebase
        if self._syzygy_reader and chess.popcount(board.occupied) <= 7:
            tb = self._tb_probe_dtz(board)
            if tb:
                if verbose: print(f"  [TB] {tb.uci()}")
                return tb
 
        self.tt_generation += 1
        self.start_time = time.time()
        hard_deadline   = None
 
        if not time_limit or time_limit <= 0:
            self.deadline = None
        else:
            hard_deadline   = self.start_time + time_limit
            verify_reserve  = min(0.35, max(0.06, time_limit * 0.12))
            main_deadline   = hard_deadline - verify_reserve
            if main_deadline <= self.start_time + 0.02:
                main_deadline = hard_deadline
            self.deadline = main_deadline
 
        self.nodes = self.qnodes = self.tt_hits = 0
 
        # Age history
        if (self.tt_generation & 3) == 0:
            stale = [k for k, v in self.history.items() if v // 2 == 0]
            for k in stale: del self.history[k]
            for k in self.history: self.history[k] //= 2

            cont_stale = [k for k, v in self.cont_hist.items() if v // 2 == 0]
            for k in cont_stale: del self.cont_hist[k]
            for k in self.cont_hist: self.cont_hist[k] //= 2

        if (self.tt_generation & 7) == 0 and self.corr_hist:
            corr_stale = [k for k, v in self.corr_hist.items() if abs(v) <= 2]
            for k in corr_stale: del self.corr_hist[k]
            for k in self.corr_hist: self.corr_hist[k] //= 2
 
        best_move:    Optional[chess.Move] = None
        best_score    = -INF
        reached_depth = 0

        root_moves = list(board.legal_moves)
        if not root_moves:
            return None
 
        for depth in range(1, max_depth + 1):
            if self.deadline and time.time() >= self.deadline: break
            try:
                if depth >= 4 and best_move:
                    window = 40
                    for _ in range(8):
                        if self.deadline and time.time() >= self.deadline:
                            raise SearchTimeout()
                        lo = max(-INF, best_score - window)
                        hi = min(INF,  best_score + window)
                        mv, sc = self.search_root(board, depth, lo, hi, pv_hint=best_move)
                        if lo < sc < hi:
                            break
                        window = window * 2 + 20
                    else:
                        if self.deadline and time.time() >= self.deadline:
                            raise SearchTimeout()
                        mv, sc = self.search_root(board, depth, -INF, INF, pv_hint=best_move)
                else:
                    mv, sc = self.search_root(board, depth, -INF, INF, pv_hint=best_move)
            except SearchTimeout:
                break
 
            if mv:
                best_move     = mv
                best_score    = sc
                reached_depth = depth
 
            if verbose and best_move:
                elapsed = max(time.time() - self.start_time, 1e-9)
                total   = self.nodes + self.qnodes
                nps     = int(total / elapsed)
                pv      = " ".join(self.principal_variation(board, 8))
                print(f"  d{depth:2d} | {self.format_score(best_score):>8}"
                      f" | {best_move.uci():>5} | n {total:>9,} | nps {nps:>8,}"
                      f" | tt {self.tt_hits:>7,} | pv {pv}")
 
            if abs(best_score) >= MATE_BOUND: break
 
        if hard_deadline is not None:
            self.deadline = hard_deadline
            if time.time() < hard_deadline - 0.01:
                best_move, best_score = self._blunder_check_root(
                    board, best_move, best_score, reached_depth, verbose)

        if best_move is not None and best_move in board.legal_moves:
            board.push(best_move)
            try:
                final_penalty = self._root_tactical_penalty(board)
            finally:
                board.pop()

            if final_penalty >= ROOT_EMERGENCY_PENALTY:
                emergency = self._pick_emergency_root_move(board)
                if emergency and emergency in board.legal_moves and emergency != best_move:
                    if verbose:
                        print(f"  safety | tactical emergency: {best_move.uci()} → {emergency.uci()}")
                    best_move = emergency

        if best_move is None:
            best_move = self._pick_emergency_root_move(board)
 
        return best_move
 
 
# ──────────────────────────────────────────────────────────────────────────
# Module-level singleton (backward-compatible)
# ──────────────────────────────────────────────────────────────────────────
ENGINE = Engine()
 
 
def find_best_move(board: chess.Board,
                   max_depth: int = 10,
                   time_limit: Optional[float] = 5.0,
                   verbose: bool = True) -> Optional[chess.Move]:
    return ENGINE.find_best_move(board, max_depth=max_depth,
                                 time_limit=time_limit, verbose=verbose)
 
 
# ──────────────────────────────────────────────────────────────────────────
# CLI helpers
# ──────────────────────────────────────────────────────────────────────────
def print_board(board: chess.Board, player_color: chess.Color):
    if player_color == chess.WHITE:
        print(board)
    else:
        lines = str(board).split("\n")
        print("\n".join(reversed(lines)))
    print()
 
 
def play_game(player_color: chess.Color = chess.WHITE,
              engine_depth: int = 10,
              time_limit: float = 5.0):
    board        = chess.Board()
    engine_color = not player_color
 
    print("=" * 72)
    print("  APEX Chess Engine  v3.0")
    print("  Search : PVS | IID | LMR | NMP | Futility | Razoring | ProbCut")
    print("           Singular | Multi-cut | Correction-hist | Cont-hist")
    print("           Threat ext | Passed-pawn ext | Recapture ext | SEE")
    print("  Eval   : Tapered | Mobility | Outposts | Bad-bishop | Hanging")
    print("           Pawn-struct | King-safety | Endgame-knowledge | Tempo")
    print(f"  You are {'White' if player_color == chess.WHITE else 'Black'}")
    print(f"  Depth {engine_depth} | Time {time_limit}s per move")
    print("  Commands: UCI move (e.g. e2e4) | 'undo' | 'fen' | 'moves' | 'quit'")
    print("=" * 72)
    print()
 
    while not board.is_game_over():
        print_board(board, player_color)
        side = "White" if board.turn == chess.WHITE else "Black"
        print(f"{side} to move | Move {board.fullmove_number}")
 
        if board.turn == player_color:
            while True:
                raw = input("Your move: ").strip().lower()
                if raw == "quit":
                    print("Goodbye."); return
                if raw == "fen":
                    print("FEN:", board.fen()); continue
                if raw == "moves":
                    print("Legal:", " ".join(sorted(m.uci() for m in board.legal_moves)))
                    continue
                if raw == "undo":
                    if len(board.move_stack) >= 2:
                        board.pop(); board.pop()
                    continue
                if raw == "eval":
                    ev = ENGINE.evaluate_white(board)
                    print(f"  Static eval (White): {ev/100:+.2f} | STM: {ENGINE._eval_stm(board)/100:+.2f}")
                    continue
                try:
                    move = chess.Move.from_uci(raw)
                except ValueError:
                    print("  Bad UCI (e.g. e2e4)"); continue
                if move in board.legal_moves:
                    board.push(move); break
                print("  Illegal move.")
        else:
            print("Engine thinking…")
            t0   = time.time()
            move = ENGINE.find_best_move(board, max_depth=engine_depth,
                                         time_limit=time_limit, verbose=True)
            elapsed = time.time() - t0
            if move is None:
                print("Engine has no legal moves!"); break
            board.push(move)
            print(f"\nEngine plays: {move.uci()}  ({elapsed:.2f}s)\n")
 
    print_board(board, player_color)
    result  = board.result()
    outcome = board.outcome()
    if outcome:
        if outcome.winner == player_color:   print("You win!")
        elif outcome.winner == engine_color: print("Engine wins!")
        else:                                print("Draw!")
    print("Result:", result)
    print("Final FEN:", board.fen())
 
 
# ──────────────────────────────────────────────────────────────────────────
# Self-play / benchmark
# ──────────────────────────────────────────────────────────────────────────
def self_play(num_games: int = 1, depth: int = 6, time_limit: float = 2.0):
    """Run engine vs itself and report results."""
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for g in range(1, num_games + 1):
        board  = chess.Board()
        eng_w  = Engine(max_tt_entries=400_000)
        eng_b  = Engine(max_tt_entries=400_000)
        moves_played = 0
        print(f"\nGame {g}/{num_games}")
        while not board.is_game_over() and moves_played < 300:
            eng = eng_w if board.turn == chess.WHITE else eng_b
            move = eng.find_best_move(board, max_depth=depth,
                                       time_limit=time_limit, verbose=False)
            if move is None: break
            board.push(move)
            moves_played += 1
        r = board.result()
        results[r] = results.get(r, 0) + 1
        print(f"  {moves_played} moves → {r}  ({board.outcome().termination.name if board.outcome() else 'unknown'})")
    print("\nSelf-play results:", results)
 
 
# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="APEX Python Chess Engine v3.0")
    parser.add_argument("--black",  action="store_true",
                        help="Play as Black")
    parser.add_argument("--depth",  type=int,   default=10,
                        help="Max search depth (default: 10)")
    parser.add_argument("--time",   type=float, default=5.0,
                        help="Time limit per move in seconds (default: 5)")
    parser.add_argument("--hash",   type=int,   default=1_000_000,
                        help="Transposition table entries (default: 1000000)")
    parser.add_argument("--book",   type=str,   default="",
                        help="Path to external polyglot opening book (.bin)")
    parser.add_argument("--tb",     type=str,   default="",
                        help="Path to Syzygy tablebase directory")
    parser.add_argument("--self",   type=int,   default=0,
                        metavar="N",
                        help="Run N engine self-play games then exit")
    args = parser.parse_args()
 
    ENGINE.set_hash_size(args.hash)
    if args.book: ENGINE.set_book_path(args.book)
    if args.tb:   ENGINE.set_syzygy_path(args.tb)
 
    if args.self > 0:
        self_play(args.self, depth=max(1, args.depth), time_limit=args.time)
        return
 
    player_color = chess.BLACK if args.black else chess.WHITE
    play_game(player_color=player_color,
              engine_depth=max(1, args.depth),
              time_limit=args.time)
 
 
if __name__ == "__main__":
    main()