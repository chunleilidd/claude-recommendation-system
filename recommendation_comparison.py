#!/usr/bin/env python3
"""
recommendation_comparison.py

Quantitatively evaluates recommendation systems (A, B, C) against a user's
past food order history.

Algorithm:
  For each order item o with recency weight w_o:
    - best_taste_score(o, S) = max over all tastes t in system S of match(t, o)
  system_score(S) = sum(w_o * best_taste_score(o, S)) / sum(w_o)

Usage:
  python recommendation_comparison.py

No third-party dependencies required (uses stdlib only).
"""

import csv
import json
import math
import os
import re
import difflib
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TasteEntry:
    rank: int
    taste_name: str
    cuisine_types: List[str]
    food_types: List[str]


@dataclass
class RecommendationSystem:
    name: str
    tastes: List[TasteEntry]


@dataclass
class OrderItem:
    item_name: str
    category_name: str
    active_date: date
    local_hour: int
    day_of_week: int
    day_type: str
    full_text: str = field(init=False)

    def __post_init__(self):
        self.full_text = f"{self.item_name} {self.category_name}"


@dataclass
class OrderMatchResult:
    item_name: str
    best_taste_name: str
    best_taste_score: float
    recency_weight: float


@dataclass
class TasteBreakdown:
    taste_name: str
    rank: int
    matched_items: List[str]
    total_weighted_score: float
    total_matched_weight: float

    @property
    def weighted_avg_score(self) -> float:
        if self.total_matched_weight == 0:
            return 0.0
        return self.total_weighted_score / self.total_matched_weight

    @property
    def match_count(self) -> int:
        return len(self.matched_items)


@dataclass
class SystemResult:
    system_name: str
    system_score: float
    orders_used: int
    taste_breakdowns: List[TasteBreakdown]
    order_results: List[OrderMatchResult]


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_md(filepath: str) -> List[RecommendationSystem]:
    """
    Parse the recommendation .md file into RecommendationSystem objects.
    Expected format per taste line:
      <rank>\ttaste = "<name>"[.]\t<json>
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    systems = []

    # Split content into per-system blocks
    system_block_re = re.compile(
        r"Recommendation System (\w+):\s*\n(.*?)(?=\nRecommendation System \w+:|\Z)",
        re.DOTALL,
    )

    # Match taste lines: rank \t taste="..." \t {...}
    taste_line_re = re.compile(
        r"^\s*(\d+)\s+taste\s*=\s*[\"']([^\"']+)[\"'][.\s]*(\{.*?\})\s*$",
        re.MULTILINE,
    )

    for sys_match in system_block_re.finditer(content):
        sys_letter = sys_match.group(1)
        block = sys_match.group(2)
        tastes = []

        for taste_match in taste_line_re.finditer(block):
            rank = int(taste_match.group(1))
            taste_name = taste_match.group(2).strip()
            json_str = taste_match.group(3).strip()

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to salvage with ast.literal_eval
                try:
                    import ast
                    data = ast.literal_eval(json_str)
                except Exception:
                    print(f"  Warning: could not parse JSON for taste '{taste_name}' in System {sys_letter}")
                    data = {}

            tastes.append(TasteEntry(
                rank=rank,
                taste_name=taste_name,
                cuisine_types=data.get("cuisine_type", []),
                food_types=data.get("food_type", []),
            ))

        if tastes:
            systems.append(RecommendationSystem(
                name=sys_letter,
                tastes=sorted(tastes, key=lambda t: t.rank),
            ))

    return systems


def parse_csv(filepath: str) -> Tuple[List[OrderItem], date]:
    """
    Parse the orders CSV into OrderItem objects.
    Uses csv.reader to handle multi-line quoted fields (e.g. DESCRIPTION).

    Column indices (0-based):
      4  = ITEM_NAME
      7  = CATEGORY_NAME
      14 = ACTIVE_DATE
      19 = LOCAL_HOUR
      20 = DAY_OF_WEEK
      21 = DAY_TYPE
    """
    orders = []

    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for row in reader:
            if len(row) < 22:
                continue
            try:
                active_date_str = row[14].strip()
                if not re.match(r"\d{4}-\d{2}-\d{2}", active_date_str):
                    continue
                active_date = date.fromisoformat(active_date_str[:10])
                local_hour_str = row[19].strip()
                if not local_hour_str.lstrip("-").isdigit():
                    continue
                local_hour = int(local_hour_str)
                day_of_week_str = row[20].strip()
                day_of_week = int(day_of_week_str) if day_of_week_str.isdigit() else 0

                orders.append(OrderItem(
                    item_name=row[4].strip(),
                    category_name=row[7].strip(),
                    active_date=active_date,
                    local_hour=local_hour,
                    day_of_week=day_of_week,
                    day_type=row[21].strip().lower(),
                ))
            except (ValueError, IndexError):
                continue

    # Infer reference date from filename, fall back to latest order date
    fname = os.path.basename(filepath)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    if date_match:
        reference_date = date.fromisoformat(date_match.group(1))
    else:
        reference_date = max(o.active_date for o in orders) if orders else date.today()

    return orders, reference_date


# ─────────────────────────────────────────────────────────────────────────────
# 2. DAY PART FILTERING
# ─────────────────────────────────────────────────────────────────────────────

DAYPART_HOURS: Dict[str, Tuple[int, int]] = {
    "breakfast": (6,  10),
    "lunch":     (11, 16),
    "afternoon": (15, 16),
    "dinner":    (17, 23),
    "all":       (0,  23),
}


def filter_orders(
    orders: List[OrderItem],
    daypart: str = "all",
    weekday_only: bool = False,
) -> List[OrderItem]:
    """Filter orders by day part and optionally by weekday."""
    if daypart not in DAYPART_HOURS:
        raise ValueError(f"Unknown daypart '{daypart}'. Choose from: {list(DAYPART_HOURS)}")

    low, high = DAYPART_HOURS[daypart]
    result = [
        o for o in orders
        if low <= o.local_hour <= high
        and (not weekday_only or o.day_type == "weekday")
    ]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. RECENCY WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────

def recency_weight(
    order_date: date,
    reference_date: date,
    decay_type: str = "exponential",
    half_life_days: float = 90.0,
) -> float:
    """
    Compute recency weight for one order.

    decay_type options:
      'exponential' : w = exp(-ln2 * age / half_life_days)   → w=1 now, w=0.5 at half_life
      'linear'      : w = max(0, 1 - age / (2*half_life))    → w=0 at 2*half_life
      'step'        : w = 1.0 if age <= half_life else 0.0   → binary window
      'none'        : w = 1.0 always
    """
    age = max(0, (reference_date - order_date).days)

    if decay_type == "exponential":
        return math.exp(-math.log(2) * age / half_life_days)
    elif decay_type == "linear":
        return max(0.0, 1.0 - age / (2.0 * half_life_days))
    elif decay_type == "step":
        return 1.0 if age <= half_life_days else 0.0
    elif decay_type == "none":
        return 1.0
    else:
        raise ValueError(f"Unknown decay_type '{decay_type}'. Choose: exponential, linear, step, none")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEXT TOKENIZATION & N-GRAM MATCHING
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "with", "and", "or", "the", "a", "an", "of", "in", "on", "at", "to",
    "for", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "this", "that", "from", "by", "as", "but", "not", "no", "so", "if",
}


def tokenize(text: str) -> List[str]:
    """
    Normalize text to lowercase ASCII tokens, removing stopwords.
    Chinese/Unicode characters are stripped (become spaces).
    """
    lowered = text.lower()
    ascii_only = re.sub(r"[^a-z0-9\s]", " ", lowered)
    tokens = ascii_only.split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def get_ngrams(tokens: List[str], n: int) -> Set[str]:
    """Return set of space-joined n-grams from a token list."""
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}


def exact_ngram_score(
    order_text: str,
    taste_terms: List[str],
    ngram_sizes: Tuple[int, ...] = (1, 2, 3),
) -> float:
    """
    Recall-oriented exact n-gram overlap.

    score = |order_ngrams ∩ taste_ngrams| / |taste_ngrams|

    Measures: what fraction of the taste's descriptive n-grams appear
    in the order text. Averaged across all specified n-gram sizes.
    Returns 0.0 if taste has no valid n-grams.
    """
    order_tokens = tokenize(order_text)
    taste_tokens = tokenize(" ".join(taste_terms))

    scores = []
    for n in ngram_sizes:
        order_grams = get_ngrams(order_tokens, n)
        taste_grams = get_ngrams(taste_tokens, n)
        if not taste_grams:
            continue
        overlap = order_grams & taste_grams
        scores.append(len(overlap) / len(taste_grams))

    return float(sum(scores) / len(scores)) if scores else 0.0


def partial_ratio(s1: str, s2: str) -> float:
    """
    Slide the shorter string over the longer and return best window similarity.
    Equivalent to rapidfuzz.fuzz.partial_ratio but uses stdlib difflib.
    Returns float in [0.0, 1.0].
    """
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if not s1 or not s2:
        return 0.0
    shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    k = len(shorter)
    best = 0.0
    for i in range(len(longer) - k + 1):
        window = longer[i: i + k]
        ratio = difflib.SequenceMatcher(None, shorter, window).ratio()
        if ratio > best:
            best = ratio
    return best


def fuzzy_taste_score(order_text: str, taste: TasteEntry) -> float:
    """
    Compute fuzzy match score between an order item and a taste entry.

    Strategy: for each food_type (and the taste_name itself), compute
    partial_ratio against the order text. Return the maximum.
    This identifies the single best food_type match.
    """
    candidates = taste.food_types + [taste.taste_name]
    candidates = [c for c in candidates if c]
    if not candidates:
        return 0.0
    return max(partial_ratio(order_text, c) for c in candidates)


def combined_score(
    order_text: str,
    taste: TasteEntry,
    ngram_sizes: Tuple[int, ...] = (1, 2, 3),
    alpha: float = 0.5,
) -> float:
    """Weighted blend: alpha * fuzzy + (1-alpha) * exact."""
    taste_terms = taste.food_types + [taste.taste_name]
    f = fuzzy_taste_score(order_text, taste)
    e = exact_ngram_score(order_text, taste_terms, ngram_sizes)
    return alpha * f + (1.0 - alpha) * e


def taste_match_score(
    order_text: str,
    taste: TasteEntry,
    match_type: str = "fuzzy",
    ngram_sizes: Tuple[int, ...] = (1, 2, 3),
) -> float:
    """Dispatch to the appropriate scoring function."""
    if match_type == "fuzzy":
        return fuzzy_taste_score(order_text, taste)
    elif match_type == "exact":
        taste_terms = taste.food_types + [taste.taste_name]
        return exact_ngram_score(order_text, taste_terms, ngram_sizes)
    elif match_type == "combined":
        return combined_score(order_text, taste, ngram_sizes)
    else:
        raise ValueError(f"Unknown match_type '{match_type}'. Choose: fuzzy, exact, combined")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SYSTEM SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_system(
    system: RecommendationSystem,
    filtered_orders: List[OrderItem],
    match_type: str,
    decay_type: str,
    half_life_days: float,
    reference_date: date,
    ngram_sizes: Tuple[int, ...] = (1, 2, 3),
) -> SystemResult:
    """
    Compute the overall quality score for one recommendation system.

    For each order item:
      1. Score it against every taste in the system.
      2. Take the best-matching taste (max score).
      3. Weight the score by the order's recency weight.

    system_score = sum(w * best_score) / sum(w)
    """
    if not filtered_orders:
        return SystemResult(
            system_name=system.name,
            system_score=0.0,
            orders_used=0,
            taste_breakdowns=[],
            order_results=[],
        )

    # Per-taste accumulators
    taste_accum: Dict[int, Dict] = {
        t.rank: {
            "taste_name": t.taste_name,
            "rank": t.rank,
            "matched_items": [],
            "total_weighted_score": 0.0,
            "total_matched_weight": 0.0,
        }
        for t in system.tastes
    }

    order_results = []
    total_weighted_score = 0.0
    total_weight = 0.0

    for order in filtered_orders:
        w = recency_weight(order.active_date, reference_date, decay_type, half_life_days)
        if w == 0.0:
            continue

        # Score every taste for this order
        taste_scores = [
            (taste, taste_match_score(order.full_text, taste, match_type, ngram_sizes))
            for taste in system.tastes
        ]

        # Best taste (tie-break: prefer lower rank = more recommended)
        best_taste, best_score = max(
            taste_scores,
            key=lambda x: (x[1], -x[0].rank),
        )

        total_weighted_score += w * best_score
        total_weight += w

        # Accumulate into per-taste breakdown
        acc = taste_accum[best_taste.rank]
        acc["matched_items"].append(order.item_name)
        acc["total_weighted_score"] += w * best_score
        acc["total_matched_weight"] += w

        order_results.append(OrderMatchResult(
            item_name=order.item_name,
            best_taste_name=best_taste.taste_name,
            best_taste_score=round(best_score, 4),
            recency_weight=round(w, 4),
        ))

    system_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

    taste_breakdowns = [
        TasteBreakdown(
            taste_name=acc["taste_name"],
            rank=acc["rank"],
            matched_items=acc["matched_items"],
            total_weighted_score=acc["total_weighted_score"],
            total_matched_weight=acc["total_matched_weight"],
        )
        for acc in sorted(taste_accum.values(), key=lambda x: -x["total_weighted_score"])
    ]

    return SystemResult(
        system_name=system.name,
        system_score=round(system_score, 6),
        orders_used=len(order_results),
        taste_breakdowns=taste_breakdowns,
        order_results=order_results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. SCENARIO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(
    systems: List[RecommendationSystem],
    all_orders: List[OrderItem],
    reference_date: date,
    scenario_label: str,
    daypart: str = "all",
    decay_type: str = "exponential",
    half_life_days: float = 90.0,
    match_type: str = "fuzzy",
    ngram_sizes: Tuple[int, ...] = (1, 2, 3),
    weekday_only: bool = False,
) -> Tuple[str, List[SystemResult]]:
    """Execute one comparison scenario across all systems."""
    filtered = filter_orders(all_orders, daypart=daypart, weekday_only=weekday_only)

    if len(filtered) < 5:
        print(f"  ⚠  Warning: only {len(filtered)} orders after filtering for '{scenario_label}'")

    results = [
        score_system(
            system=sys,
            filtered_orders=filtered,
            match_type=match_type,
            decay_type=decay_type,
            half_life_days=half_life_days,
            reference_date=reference_date,
            ngram_sizes=ngram_sizes,
        )
        for sys in systems
    ]

    results.sort(key=lambda r: -r.system_score)
    return scenario_label, results


# ─────────────────────────────────────────────────────────────────────────────
# 7. OUTPUT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def print_results(
    scenario_label: str,
    results: List[SystemResult],
    daypart: str,
    decay_type: str,
    half_life_days: float,
    match_type: str,
    ngram_sizes: Tuple[int, ...],
    verbose: bool = True,
) -> None:
    """Print a formatted comparison table for one scenario."""
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  {scenario_label}")
    orders_used = results[0].orders_used if results else 0
    print(
        f"  Orders: {orders_used} | Daypart: {daypart} | "
        f"Decay: {decay_type}(½={half_life_days}d) | Match: {match_type} | "
        f"N-grams: {ngram_sizes}"
    )
    print(sep)

    # System ranking table
    print(f"\n  {'Rank':<6} {'System':<8} {'Score':>8}   {'Orders':>7}")
    print(f"  {'-'*4:<6} {'-'*6:<8} {'-'*8:>8}   {'-'*6:>7}")
    for i, r in enumerate(results, 1):
        print(f"  {i:<6} {r.system_name:<8} {r.system_score:>8.4f}   {r.orders_used:>7}")

    if not verbose:
        return

    # Per-taste breakdown for each system
    for r in results:
        print(f"\n  ── Taste Breakdown: System {r.system_name} (score={r.system_score:.4f}) ──")
        print(f"  {'Rank':<5} {'Taste':<40} {'Matches':>7}  {'Wtd Avg':>8}  Top Matched Items")
        print(f"  {'-'*5} {'-'*40} {'-'*7}  {'-'*8}  {'-'*30}")

        for tb in r.taste_breakdowns:
            sample = ", ".join(tb.matched_items[:3])
            if len(tb.matched_items) > 3:
                sample += f" (+{len(tb.matched_items)-3} more)"
            score_str = f"{tb.weighted_avg_score:.4f}" if tb.match_count > 0 else "  —   "
            print(
                f"  {tb.rank:<5} {tb.taste_name:<40} {tb.match_count:>7}  "
                f"{score_str:>8}  {sample}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN — 4 SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "scenario_label": "Scenario 1: Lunch | Exponential decay (½=90d) | Fuzzy",
        "daypart": "lunch",
        "decay_type": "exponential",
        "half_life_days": 90.0,
        "match_type": "fuzzy",
        "ngram_sizes": (1, 2, 3),
        "weekday_only": False,
    },
    {
        "scenario_label": "Scenario 2: Lunch | Step decay (window=90d) | Exact n-gram",
        "daypart": "lunch",
        "decay_type": "step",
        "half_life_days": 90.0,
        "match_type": "exact",
        "ngram_sizes": (1, 2, 3),
        "weekday_only": False,
    },
    {
        "scenario_label": "Scenario 3: Lunch | No recency | Fuzzy",
        "daypart": "lunch",
        "decay_type": "none",
        "half_life_days": 90.0,
        "match_type": "fuzzy",
        "ngram_sizes": (1, 2, 3),
        "weekday_only": False,
    },
    {
        "scenario_label": "Scenario 4: Dinner | Exponential decay (½=90d) | Fuzzy",
        "daypart": "dinner",
        "decay_type": "exponential",
        "half_life_days": 90.0,
        "match_type": "fuzzy",
        "ngram_sizes": (1, 2, 3),
        "weekday_only": False,
    },
]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_path  = os.path.join(base_dir, "sample_recommendation.md")
    csv_path = os.path.join(base_dir, "Chunlei-orders-2026-03-04.csv")

    # ── Load data ──
    print("Loading recommendation systems...")
    systems = parse_md(md_path)
    print(f"  Loaded {len(systems)} systems: {[s.name for s in systems]}")
    for s in systems:
        print(f"    System {s.name}: {len(s.tastes)} tastes")

    print("\nLoading order history...")
    orders, reference_date = parse_csv(csv_path)
    print(f"  Loaded {len(orders)} order items")
    print(f"  Date range: {min(o.active_date for o in orders)} → {max(o.active_date for o in orders)}")
    print(f"  Reference date (for recency): {reference_date}")

    # ── Run scenarios ──
    all_scenario_results = []
    for cfg in SCENARIOS:
        label, results = run_scenario(systems, orders, reference_date, **cfg)
        all_scenario_results.append((label, results, cfg))
        print_results(
            scenario_label=label,
            results=results,
            daypart=cfg["daypart"],
            decay_type=cfg["decay_type"],
            half_life_days=cfg["half_life_days"],
            match_type=cfg["match_type"],
            ngram_sizes=cfg["ngram_sizes"],
            verbose=True,
        )

    # ── Overall summary ──
    print(f"\n{'═'*70}")
    print("  OVERALL SUMMARY — Which system wins across scenarios?")
    print(f"{'═'*70}")
    win_counts = {s.name: 0 for s in systems}
    score_totals = {s.name: 0.0 for s in systems}

    for _, results, _ in all_scenario_results:
        winner = results[0].system_name
        win_counts[winner] += 1
        for r in results:
            score_totals[r.system_name] += r.system_score

    print(f"\n  {'System':<10} {'Wins':>6}  {'Avg Score':>10}")
    print(f"  {'-'*10} {'-'*6}  {'-'*10}")
    n_scenarios = len(SCENARIOS)
    sorted_systems = sorted(win_counts.items(), key=lambda x: (-x[1], -score_totals[x[0]]))
    for sys_name, wins in sorted_systems:
        avg = score_totals[sys_name] / n_scenarios
        print(f"  System {sys_name:<4}  {wins:>5}/4   {avg:>10.4f}")

    best_system = sorted_systems[0][0]
    print(f"\n  ✓ Best overall system: System {best_system}")
    print()


if __name__ == "__main__":
    main()
