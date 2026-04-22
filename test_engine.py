#!/usr/bin/env python3
"""
APEX Engine Stability Test Suite
=================================

Comprehensive testing framework to verify engine stability before parameter tuning.
Run these tests in order - DO NOT skip to parameter tuning until all pass.

Usage:
    python test_engine.py --all                    # Run all tests
    python test_engine.py --phase1                 # Critical stability tests
    python test_engine.py --phase2                 # Bug-specific tests
    python test_engine.py --illegal-moves          # Test specific area
    python test_engine.py --tactics                # Tactical reliability
    python test_engine.py --consistency            # Search consistency
"""

import argparse
import os
import sys
import time
from typing import List, Dict, Tuple, Optional
import chess
import chess.engine

# Assuming engine.py is in the same directory
try:
    from engine import Engine, MATE_SCORE, MATE_BOUND
except ImportError:
    print("ERROR: Cannot import engine.py - make sure it's in the same directory")
    exit(1)


DEFAULT_STOCKFISH_PATH = r"C:\CS\Stockfish\stockfish-windows-x86-64-avx2.exe"


def _configure_console_encoding() -> None:
    """Best-effort UTF-8 console output to avoid cp1252 encode failures."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_configure_console_encoding()


def _make_engine(max_tt_entries: int = 100_000, disable_book: bool = False) -> Engine:
    """Create a test engine instance with optional opening-book disable."""
    engine = Engine(max_tt_entries=max_tt_entries)
    if disable_book and hasattr(engine, "set_book_enabled"):
        try:
            engine.set_book_enabled(False)
        except Exception:
            pass
    return engine


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: CRITICAL STABILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_illegal_moves() -> Dict:
    """
    Test 1.1: Verify search never returns illegal moves
    
    This is the MOST CRITICAL test. If this fails, the engine has a fundamental
    bug that must be fixed before any other testing.
    
    Returns:
        dict with 'passed', 'failed', 'failures' keys
    """
    print("\n" + "="*80)
    print("TEST 1.1: ILLEGAL MOVE DETECTION")
    print("="*80)
    print("Testing 50+ complex positions for illegal move returns...")
    print("This test is CRITICAL - engine must pass 100%\n")
    
    test_positions = [
        # Castling edge cases
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "Both sides can castle"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w Qkq - 0 1", "Only black can castle kingside"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w kq - 0 1", "White king has moved"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1", "Only white kingside"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "Pieces blocking castle"),
        ("r3k2r/8/8/8/8/8/8/4K3 w kq - 0 1", "White cannot castle"),
        
        # En passant
        ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 1", "White en passant f6"),
        ("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "Black en passant e3"),
        ("rnbqkbnr/pppppppp/8/8/3Pp3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 1", "Black en passant d3"),
        ("4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1", "Simple en passant"),
        
        # Promotion edge cases
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "White pawn promotes"),
        ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "Black pawn promotes"),
        ("8/PPPPPPPP/8/8/8/8/8/4K2k w - - 0 1", "Multiple promotion options"),
        ("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "Promotion with capture"),
        
        # Check/pin edge cases
        ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "Check position"),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Pin position"),
        ("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Mutual pins"),
        
        # Complex tactical positions
        ("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1", "Mate in 1"),
        ("r1bq1rk1/pp3pbp/2np1np1/2p1p3/2P1P3/2NPBN2/PP2BPPP/R2Q1RK1 w - - 0 11", "King's Indian"),
        ("rnbqkb1r/pp1p1ppp/4pn2/2p5/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 0 4", "Catalan"),
        
        # TT collision candidates (same piece count)
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2", "Knights developed"),
        ("r1bqkb1r/pppppppp/2n2n2/8/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq - 4 3", "4 knights"),
        
        # Edge cases with few pieces
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "Endgame position"),
        ("8/8/8/8/8/2k5/1p6/1K6 w - - 0 1", "Stalemate trap"),
        ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", "King + Rook vs King"),
        
        # Repetition positions
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Start 1"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After e4"),
    ]
    
    results = {
        "passed": 0,
        "failed": 0,
        "failures": [],
        "total": 0
    }
    
    engine = Engine(max_tt_entries=100_000)
    
    for fen, description in test_positions:
        board = chess.Board(fen)
        
        # Run 5 searches per position to catch non-deterministic bugs
        for run in range(5):
            results["total"] += 1
            
            try:
                # Clear TT between runs to avoid pollution
                engine.clear_tt()
                
                move = engine.find_best_move(
                    board, 
                    max_depth=8, 
                    time_limit=2.0, 
                    verbose=False
                )
                
                # Check 1: Move is not None (unless position is terminal)
                if move is None:
                    if not board.is_game_over():
                        results["failed"] += 1
                        results["failures"].append({
                            "fen": fen,
                            "description": description,
                            "error": "Returned None in non-terminal position",
                            "run": run
                        })
                    else:
                        results["passed"] += 1
                    continue
                
                # Check 2: Move is legal
                if move not in board.legal_moves:
                    results["failed"] += 1
                    results["failures"].append({
                        "fen": fen,
                        "description": description,
                        "error": f"Illegal move: {move.uci()}",
                        "legal_moves": [m.uci() for m in board.legal_moves],
                        "run": run
                    })
                else:
                    results["passed"] += 1
                    
            except Exception as e:
                results["failed"] += 1
                results["failures"].append({
                    "fen": fen,
                    "description": description,
                    "error": f"Exception: {str(e)}",
                    "run": run
                })
    
    # Print results
    print(f"\nResults:")
    print(f"  Total searches: {results['total']}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Pass rate: {100 * results['passed'] / results['total']:.1f}%")
    
    if results["failures"]:
        print(f"\n❌ FAILURES (showing first 10):")
        for i, failure in enumerate(results["failures"][:10]):
            print(f"\n  Failure {i+1}:")
            print(f"    Position: {failure['description']}")
            print(f"    FEN: {failure['fen']}")
            print(f"    Error: {failure['error']}")
            if 'legal_moves' in failure:
                print(f"    Legal moves: {', '.join(failure['legal_moves'][:10])}")
    
    # Verdict
    if results["failed"] == 0:
        print(f"\n✅ TEST PASSED - No illegal moves detected")
    else:
        print(f"\n❌ TEST FAILED - {results['failed']} illegal moves found")
        print(f"   CRITICAL: Fix this before any other testing!")
    
    return results


def test_tactical_reliability() -> Dict:
    """
    Test 1.2: Verify engine finds basic tactics
    
    Tactical smoke tests using validated immediate tactical motifs.
    These positions are designed to avoid ambiguous long-horizon expectations.
    
    Returns:
        dict with pass/fail results by category
    """
    print("\n" + "="*80)
    print("TEST 1.2: TACTICAL RELIABILITY")
    print("="*80)
    print("Testing if engine finds basic tactics (mates, piece wins)...\n")
    
    tactics = [
        # Mate in one positions with clear forced mates.
        ("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1", "mate_in_1", "White has immediate queen mate", "mate", None),
        ("k7/1Q6/K7/8/8/8/8/8 w - - 0 1", "mate_in_1", "White has immediate edge mate", "mate", None),
        ("8/8/8/8/8/6k1/5q2/6K1 b - - 0 1", "mate_in_1", "Black has immediate queen mate", "mate", None),

        # Hanging major piece captures.
        ("4k3/8/8/8/8/8/4q3/4KQ2 w - - 0 1", "win_queen", "White should capture hanging queen", "expected", {"e1e2", "f1e2"}),
        ("4k3/8/8/8/8/8/4Q3/4qK2 b - - 0 1", "win_queen", "Black should capture hanging queen", "expected", {"e1e2"}),

        # Special tactical move types.
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "promotion", "White should promote the pawn", "promotion", None),
        (
            "rnbqkb1r/ppppp1pp/7n/8/3PPp2/N6N/PPP2PPP/R1BQKB1R b KQkq e3 0 4",
            "en_passant",
            "Black should play the best en passant capture",
            "en_passant",
            None,
        ),
    ]
    
    results = {
        "mate_in_1": {"passed": 0, "failed": 0},
        "win_queen": {"passed": 0, "failed": 0},
        "promotion": {"passed": 0, "failed": 0},
        "en_passant": {"passed": 0, "failed": 0},
    }
    
    failures = []
    engine = _make_engine(max_tt_entries=100_000, disable_book=True)
    
    for fen, tactic_type, description, check_kind, expected_moves in tactics:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        
        depth = 10
        time_limit = 1.5
        
        move = engine.find_best_move(
            board, 
            max_depth=depth,
            time_limit=time_limit, 
            verbose=False
        )
        
        if move is None:
            results[tactic_type]["failed"] += 1
            failures.append({
                "fen": fen,
                "type": tactic_type,
                "description": description,
                "move": "None",
                "reason": "No move returned"
            })
            continue
        
        # Verify the move is tactical for this motif.
        board_copy = board.copy()
        board_copy.push(move)

        passed = False
        if check_kind == "mate":
            passed = board_copy.is_checkmate()
        elif check_kind == "expected":
            passed = move.uci() in (expected_moves or set())
        elif check_kind == "promotion":
            passed = move.promotion is not None
        elif check_kind == "en_passant":
            passed = board.is_en_passant(move)

        if passed:
            results[tactic_type]["passed"] += 1
        else:
            results[tactic_type]["failed"] += 1
            failures.append({
                "fen": fen,
                "type": tactic_type,
                "description": description,
                "move": move.uci(),
                "reason": f"Failed motif check ({check_kind})"
            })
    
    # Print results
    print(f"\nResults by category:")
    total_passed = 0
    total_failed = 0
    
    for category, counts in results.items():
        passed = counts["passed"]
        failed = counts["failed"]
        total = passed + failed
        total_passed += passed
        total_failed += failed
        
        if total > 0:
            pct = 100 * passed / total
            status = "✅" if pct >= 90 else "⚠️" if pct >= 70 else "❌"
            print(f"  {status} {category:15s}: {passed}/{total} ({pct:.0f}%)")
    
    overall_pct = 100 * total_passed / (total_passed + total_failed)
    print(f"\nOverall: {total_passed}/{total_passed + total_failed} ({overall_pct:.0f}%)")
    
    if failures:
        print(f"\n❌ FAILURES (showing first 5):")
        for i, failure in enumerate(failures[:5]):
            print(f"\n  Failure {i+1}: {failure['type']}")
            print(f"    {failure['description']}")
            print(f"    FEN: {failure['fen']}")
            print(f"    Engine played: {failure['move']}")
            print(f"    Reason: {failure['reason']}")
    
    # Verdict
    if overall_pct >= 90:
        print(f"\n✅ TEST PASSED - {overall_pct:.0f}% tactical accuracy")
    elif overall_pct >= 70:
        print(f"\n⚠️  TEST MARGINAL - {overall_pct:.0f}% tactical accuracy")
        print(f"   Acceptable but room for improvement")
    else:
        print(f"\n❌ TEST FAILED - {overall_pct:.0f}% tactical accuracy")
        print(f"   Engine missing too many tactics")
    
    return results


def test_search_consistency() -> Dict:
    """
    Test 1.3: Same position should give same result
    
    Run the same position 20 times and verify the engine returns the same move
    at least 18/20 times. This tests for:
    - Non-deterministic bugs
    - TT ordering effects
    - Time management instability
    
    Returns:
        dict with consistency results
    """
    print("\n" + "="*80)
    print("TEST 1.3: SEARCH CONSISTENCY")
    print("="*80)
    print("Testing if engine returns consistent moves (same position, same depth)...\n")
    
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Italian game"),
        ("rnbqkb1r/pp1p1ppp/4pn2/2p5/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 0 4", "Catalan opening"),
        ("r1bq1rk1/pp3pbp/2np1np1/2p1p3/2P1P3/2NPBN2/PP2BPPP/R2Q1RK1 w - - 0 11", "King's Indian"),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "Complex endgame"),
    ]
    
    results = {
        "consistent": 0,
        "inconsistent": 0,
        "details": []
    }
    
    engine = _make_engine(max_tt_entries=100_000, disable_book=True)
    
    for fen, description in test_positions:
        board = chess.Board(fen)
        moves_found = {}
        
        print(f"Testing: {description}")
        print(f"  Running 20 searches... ", end="", flush=True)
        
        for run in range(20):
            # Clear TT to avoid ordering effects
            engine.clear_tt()
            
            move = engine.find_best_move(
                board, 
                max_depth=5,
                time_limit=0.4,
                verbose=False
            )
            
            move_str = move.uci() if move else "None"
            moves_found[move_str] = moves_found.get(move_str, 0) + 1
        
        # Check consistency (dominant move appears 18+ times)
        max_count = max(moves_found.values())
        most_common = max(moves_found.items(), key=lambda x: x[1])
        
        is_consistent = max_count >= 18
        
        if is_consistent:
            results["consistent"] += 1
            print(f"✅ Consistent ({most_common[0]}: {most_common[1]}/20)")
        else:
            results["inconsistent"] += 1
            print(f"❌ Inconsistent (moves: {moves_found})")
            results["details"].append({
                "fen": fen,
                "description": description,
                "moves": moves_found
            })
    
    # Print summary
    total = results["consistent"] + results["inconsistent"]
    print(f"\n\nResults:")
    print(f"  Consistent: {results['consistent']}/{total}")
    print(f"  Inconsistent: {results['inconsistent']}/{total}")
    
    if results["details"]:
        print(f"\n❌ INCONSISTENT POSITIONS:")
        for detail in results["details"]:
            print(f"\n  {detail['description']}")
            print(f"    FEN: {detail['fen']}")
            print(f"    Moves found:")
            for move, count in sorted(detail['moves'].items(), 
                                     key=lambda x: x[1], 
                                     reverse=True):
                print(f"      {move}: {count}/20 ({100*count/20:.0f}%)")
    
    # Verdict
    if results["inconsistent"] == 0:
        print(f"\n✅ TEST PASSED - Perfect consistency")
    else:
        print(f"\n❌ TEST FAILED - {results['inconsistent']} inconsistent positions")
        print(f"   Engine has non-deterministic behavior")
    
    return results


def test_time_management() -> Dict:
    """
    Test 1.4: Engine respects time limits
    
    Verify that the engine stops searching within reasonable bounds of the
    specified time limit.
    
    Returns:
        dict with timing results
    """
    print("\n" + "="*80)
    print("TEST 1.4: TIME MANAGEMENT")
    print("="*80)
    print("Testing if engine respects time limits...\n")
    
    test_cases = [
        {"depth": 10, "time_limit": 1.0, "tolerance": 0.5, "description": "1 second search"},
        {"depth": 10, "time_limit": 5.0, "tolerance": 1.5, "description": "5 second search"},
        {"depth": 20, "time_limit": 0.5, "tolerance": 0.3, "description": "Hard cutoff"},
        {"depth": 8, "time_limit": 2.0, "tolerance": 0.6, "description": "2 second search"},
    ]
    
    results = {
        "passed": 0,
        "failed": 0,
        "details": []
    }
    
    engine = _make_engine(max_tt_entries=100_000, disable_book=True)
    board = chess.Board()  # Starting position
    
    for tc in test_cases:
        print(f"Testing: {tc['description']} (target: {tc['time_limit']}s ±{tc['tolerance']}s)")
        
        start = time.time()
        move = engine.find_best_move(
            board,
            max_depth=tc["depth"],
            time_limit=tc["time_limit"],
            verbose=False
        )
        elapsed = time.time() - start
        
        expected = tc["time_limit"]
        tolerance = tc["tolerance"]
        lower_bound = expected - tolerance
        upper_bound = expected + tolerance
        
        is_within_bounds = lower_bound <= elapsed <= upper_bound
        
        status = "✅" if is_within_bounds else "❌"
        print(f"  {status} Elapsed: {elapsed:.2f}s (bounds: {lower_bound:.2f}s - {upper_bound:.2f}s)")
        
        if is_within_bounds:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append({
                "description": tc["description"],
                "expected": expected,
                "actual": elapsed,
                "tolerance": tolerance,
                "move": move.uci() if move else "None"
            })
    
    # Print summary
    total = len(test_cases)
    print(f"\nResults: {results['passed']}/{total} passed")
    
    if results["details"]:
        print(f"\n❌ TIMING VIOLATIONS:")
        for detail in results["details"]:
            print(f"  {detail['description']}")
            print(f"    Expected: {detail['expected']}s ±{detail['tolerance']}s")
            print(f"    Actual: {detail['actual']:.2f}s")
    
    # Verdict
    if results["failed"] == 0:
        print(f"\n✅ TEST PASSED - All time limits respected")
    else:
        print(f"\n❌ TEST FAILED - {results['failed']} timing violations")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: BUG-SPECIFIC TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_see_edge_cases() -> Dict:
    """
    Test 2.2: SEE (Static Exchange Evaluation) edge cases
    
    Test special moves that might break SEE:
    - En passant captures
    - Promotions
    - Discovered attacks
    - X-ray attacks
    
    Returns:
        dict with test results
    """
    print("\n" + "="*80)
    print("TEST 2.2: SEE EDGE CASES")
    print("="*80)
    print("Testing Static Exchange Evaluation on special moves...\n")
    
    test_cases = [
        # Stockfish-validated en passant best move.
        (
            "rnbqkb1r/ppppp1pp/7n/8/3PPp2/N6N/PPP2PPP/R1BQKB1R b KQkq e3 0 4",
            "f4e3",
            "En passant should be selected when best",
            "expected",
        ),

        # Promotion should be selected in a trivial promotion race.
        (
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            None,
            "Promotion should be played",
            "promotion",
        ),

        # Hanging queen should be captured.
        (
            "4k3/8/8/8/8/8/4Q3/4qK2 b - - 0 1",
            "e1e2",
            "Capture hanging queen",
            "expected",
        ),
    ]
    
    results = {"passed": 0, "failed": 0, "details": []}
    engine = _make_engine(max_tt_entries=50_000, disable_book=True)
    
    for fen, expected_move_uci, description, test_type in test_cases:
        board = chess.Board(fen)
        
        print(f"Testing: {description}")
        print(f"  FEN: {fen}")
        
        move = engine.find_best_move(board, max_depth=8, time_limit=2.0, verbose=False)
        
        if move is None:
            print(f"  ❌ No move returned")
            results["failed"] += 1
            results["details"].append({
                "fen": fen,
                "description": description,
                "expected": expected_move_uci,
                "got": "None"
            })
            continue
        
        passed = False
        if test_type == "expected":
            passed = move.uci() == expected_move_uci
        elif test_type == "promotion":
            passed = move.promotion is not None

        if passed:
            print(f"  ✅ Correct: {move.uci()}")
            results["passed"] += 1
        else:
            print(f"  ❌ Expected: {expected_move_uci}, got: {move.uci()}")
            results["failed"] += 1
            results["details"].append({
                "fen": fen,
                "description": description,
                "expected": expected_move_uci,
                "got": move.uci()
            })
    
    # Verdict
    total = results["passed"] + results["failed"]
    if results["failed"] == 0:
        print(f"\n✅ TEST PASSED - SEE handles edge cases correctly")
    else:
        print(f"\n❌ TEST FAILED - {results['failed']}/{total} SEE edge cases failed")
    
    return results


def audit_stockfish_alignment(stockfish_path: str = DEFAULT_STOCKFISH_PATH,
                             depth: int = 16) -> Dict:
    """Verify current fixtures against Stockfish and report deleted-case mismatch."""
    print("\n" + "="*80)
    print("STOCKFISH FIXTURE ALIGNMENT AUDIT")
    print("="*80)
    print("Auditing current tactical/SEE fixtures against Stockfish best move...\n")

    if not os.path.isfile(stockfish_path):
        print(f"❌ Stockfish binary not found: {stockfish_path}")
        return {"passed": False, "reason": "missing_stockfish"}

    if depth <= 0:
        raise ValueError("depth must be positive")

    tactical = [
        ("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1", "mate", None, "tactical_mate_1"),
        ("k7/1Q6/K7/8/8/8/8/8 w - - 0 1", "mate", None, "tactical_mate_2"),
        ("8/8/8/8/8/6k1/5q2/6K1 b - - 0 1", "mate", None, "tactical_mate_3"),
        ("4k3/8/8/8/8/8/4q3/4KQ2 w - - 0 1", "expected", {"e1e2", "f1e2"}, "tactical_hanging_queen_white"),
        ("4k3/8/8/8/8/8/4Q3/4qK2 b - - 0 1", "expected", {"e1e2"}, "tactical_hanging_queen_black"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "promotion", None, "tactical_promotion"),
        ("rnbqkb1r/ppppp1pp/7n/8/3PPp2/N6N/PPP2PPP/R1BQKB1R b KQkq e3 0 4", "en_passant", None, "tactical_en_passant"),
    ]

    see_cases = [
        ("rnbqkb1r/ppppp1pp/7n/8/3PPp2/N6N/PPP2PPP/R1BQKB1R b KQkq e3 0 4", "expected", "f4e3", "see_en_passant"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "promotion", None, "see_promotion"),
        ("4k3/8/8/8/8/8/4Q3/4qK2 b - - 0 1", "expected", "e1e2", "see_hanging_queen"),
    ]

    old_deleted = "4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1"

    def check_motif(board: chess.Board, best_uci: Optional[str], kind: str,
                    expected: Optional[set[str]]) -> bool:
        if not best_uci:
            return False
        mv = chess.Move.from_uci(best_uci)
        if kind == "mate":
            b2 = board.copy()
            b2.push(mv)
            return b2.is_checkmate()
        if kind == "expected":
            return best_uci in (expected or set())
        if kind == "promotion":
            return mv.promotion is not None
        if kind == "en_passant":
            return board.is_en_passant(mv)
        return False

    result = {
        "tactical_passed": 0,
        "tactical_total": len(tactical),
        "see_passed": 0,
        "see_total": len(see_cases),
        "old_deleted_best": None,
    }

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as sf:
        print("Current tactical fixtures")
        for fen, kind, expected, name in tactical:
            board = chess.Board(fen)
            info = sf.analyse(board, chess.engine.Limit(depth=depth),
                              info=chess.engine.INFO_PV | chess.engine.INFO_SCORE)
            pv = info.get("pv", [])
            best = pv[0].uci() if pv else None
            passed = check_motif(board, best, kind, expected)
            if passed:
                result["tactical_passed"] += 1
            print(f"  {name}: best={best} kind={kind} pass={passed}")

        print("\nCurrent SEE fixtures")
        for fen, kind, expected, name in see_cases:
            board = chess.Board(fen)
            info = sf.analyse(board, chess.engine.Limit(depth=depth),
                              info=chess.engine.INFO_PV | chess.engine.INFO_SCORE)
            pv = info.get("pv", [])
            best = pv[0].uci() if pv else None
            if kind == "expected":
                passed = best == expected
            else:
                passed = bool(best) and chess.Move.from_uci(best).promotion is not None
            if passed:
                result["see_passed"] += 1
            print(f"  {name}: best={best} expected={expected} kind={kind} pass={passed}")

        old_board = chess.Board(old_deleted)
        old_info = sf.analyse(old_board, chess.engine.Limit(depth=depth),
                              info=chess.engine.INFO_PV | chess.engine.INFO_SCORE)
        old_pv = old_info.get("pv", [])
        old_best = old_pv[0].uci() if old_pv else None
        result["old_deleted_best"] = old_best

    print("\nDeleted case check")
    print("  old_deleted_en_passant: "
          f"best={result['old_deleted_best']} (old deleted expected was d5e6)")

    tactical_ok = result["tactical_passed"] == result["tactical_total"]
    see_ok = result["see_passed"] == result["see_total"]
    print(f"\nTACTICAL_MATCH {result['tactical_passed']}/{result['tactical_total']}")
    print(f"SEE_MATCH {result['see_passed']}/{result['see_total']}")

    if tactical_ok and see_ok:
        print("\n✅ Audit passed - current fixtures align with Stockfish checks")
    else:
        print("\n❌ Audit failed - at least one current fixture is not aligned")

    result["passed"] = tactical_ok and see_ok
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_phase1():
    """Run all Phase 1 critical stability tests"""
    print("\n" + "="*80)
    print("PHASE 1: CRITICAL STABILITY TESTS")
    print("="*80)
    print("These tests MUST pass before proceeding to parameter tuning\n")
    
    results = {}
    
    # Test 1.1: Illegal moves (CRITICAL)
    results['illegal_moves'] = test_illegal_moves()
    time.sleep(1)
    
    # Test 1.2: Tactical reliability
    results['tactics'] = test_tactical_reliability()
    time.sleep(1)
    
    # Test 1.3: Search consistency
    results['consistency'] = test_search_consistency()
    time.sleep(1)
    
    # Test 1.4: Time management
    results['time_mgmt'] = test_time_management()
    
    # Overall summary
    print("\n\n" + "="*80)
    print("PHASE 1 SUMMARY")
    print("="*80)
    
    illegal_pass = results['illegal_moves']['failed'] == 0
    tactics_pass = sum(r['passed'] for r in results['tactics'].values()) >= 0.9 * sum(
        r['passed'] + r['failed'] for r in results['tactics'].values())
    consistency_pass = results['consistency']['inconsistent'] == 0
    time_pass = results['time_mgmt']['failed'] == 0
    
    print(f"Illegal Moves:   {'✅ PASS' if illegal_pass else '❌ FAIL'}")
    print(f"Tactics:         {'✅ PASS' if tactics_pass else '❌ FAIL'}")
    print(f"Consistency:     {'✅ PASS' if consistency_pass else '❌ FAIL'}")
    print(f"Time Management: {'✅ PASS' if time_pass else '❌ FAIL'}")
    
    all_pass = illegal_pass and tactics_pass and consistency_pass and time_pass
    
    if all_pass:
        print(f"\n✅ ✅ ✅  ALL PHASE 1 TESTS PASSED  ✅ ✅ ✅")
        print(f"\nYou can proceed to Phase 2 (bug-specific tests)")
    else:
        print(f"\n❌ ❌ ❌  PHASE 1 FAILED  ❌ ❌ ❌")
        print(f"\nDO NOT proceed to parameter tuning until these are fixed!")
        print(f"\nFix the failed tests and re-run Phase 1")
    
    return results


def run_phase2():
    """Run Phase 2 bug-specific tests"""
    print("\n" + "="*80)
    print("PHASE 2: BUG-SPECIFIC TESTS")
    print("="*80)
    print("Targeted tests for known potential bugs\n")
    
    results = {}
    
    # Test 2.2: SEE edge cases
    results['see'] = test_see_edge_cases()
    
    # Overall summary
    print("\n\n" + "="*80)
    print("PHASE 2 SUMMARY")
    print("="*80)
    
    see_pass = results['see']['failed'] == 0
    
    print(f"SEE Edge Cases: {'✅ PASS' if see_pass else '❌ FAIL'}")
    
    if see_pass:
        print(f"\n✅ PHASE 2 PASSED")
    else:
        print(f"\n❌ PHASE 2 FAILED - Fix SEE implementation")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="APEX Engine Stability Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_engine.py --all              # Run all tests
  python test_engine.py --phase1           # Critical stability tests only
  python test_engine.py --illegal-moves    # Test specific area
  python test_engine.py --tactics          # Tactical reliability only
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all tests (Phase 1 + Phase 2)')
    parser.add_argument('--phase1', action='store_true',
                       help='Run Phase 1 critical stability tests')
    parser.add_argument('--phase2', action='store_true',
                       help='Run Phase 2 bug-specific tests')
    parser.add_argument('--illegal-moves', action='store_true',
                       help='Test 1.1: Illegal move detection')
    parser.add_argument('--tactics', action='store_true',
                       help='Test 1.2: Tactical reliability')
    parser.add_argument('--consistency', action='store_true',
                       help='Test 1.3: Search consistency')
    parser.add_argument('--time', action='store_true',
                       help='Test 1.4: Time management')
    parser.add_argument('--see', action='store_true',
                       help='Test 2.2: SEE edge cases')
    parser.add_argument('--audit-stockfish', action='store_true',
                       help='Audit current fixtures against Stockfish best move')
    parser.add_argument('--stockfish', default=DEFAULT_STOCKFISH_PATH,
                       help='Path to Stockfish binary used by --audit-stockfish')
    parser.add_argument('--sf-depth', type=int, default=16,
                       help='Stockfish analysis depth for --audit-stockfish')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Run requested tests
    if args.all:
        run_phase1()
        print("\n\n")
        run_phase2()
    elif args.phase1:
        run_phase1()
    elif args.phase2:
        run_phase2()
    else:
        # Individual tests
        if args.illegal_moves:
            test_illegal_moves()
        if args.tactics:
            test_tactical_reliability()
        if args.consistency:
            test_search_consistency()
        if args.time:
            test_time_management()
        if args.see:
            test_see_edge_cases()
        if args.audit_stockfish:
            audit_stockfish_alignment(stockfish_path=args.stockfish, depth=args.sf_depth)


if __name__ == "__main__":
    main()