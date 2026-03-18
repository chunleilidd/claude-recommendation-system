#!/usr/bin/env python3
"""
similar_title_analysis.py

Finds similar carousel titles within each <consumer_id, daypart> pair
from the Snowflake table and reports the number of distinct title groups.

Algorithm:
  - For each <consumer_id, daypart> pair, compute pairwise cosine similarity
    between EMBEDDING_256 vectors (256-dim title embeddings stored as VARIANT).
  - Group titles into clusters using union-find (connected components).
  - Report:
    1. Distribution of "number of distinct title groups" per pair.
    2. Sample <consumer_id, daypart> pairs where a cluster has > CLUSTER_SIZE_THRESHOLD
       similar titles.

Usage:
  python similar_title_analysis.py

Environment variables for Snowflake connection:
  SNOWFLAKE_ACCOUNT    (required)
  SNOWFLAKE_USER       (required)
  SNOWFLAKE_PASSWORD   (required)
  SNOWFLAKE_WAREHOUSE  (optional)
  SNOWFLAKE_DATABASE   (default: PRODDB)
  SNOWFLAKE_SCHEMA     (default: PUBLIC)
  SNOWFLAKE_ROLE       (optional)
"""

import collections
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import snowflake.connector

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TABLE = (
    "proddb.public"
    ".genai_carousels_orders_v17_gemini_2_5_pro_prompt0"
    "_all_daypart_title_2_9_employees_emb_union"
)

# Titles whose EMBEDDING_256 cosine similarity >= this are considered "similar".
# Cosine similarity in embedding space: 1.0 = identical, 0.0 = orthogonal.
# 0.90 is a tight threshold; lower to 0.85 to catch broader semantic overlap.
SIMILARITY_THRESHOLD = 0.80

# Sample at most this many distinct consumers (set to None for the full table).
SAMPLE_CONSUMERS = None

# Report pairs where at least one cluster is larger than this.
CLUSTER_SIZE_THRESHOLD = 3

# Max example pairs to print.
MAX_EXAMPLES = 15


# ─────────────────────────────────────────────────────────────────────────────
# SNOWFLAKE CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_connection() -> snowflake.connector.SnowflakeConnection:
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
        database=os.environ.get("SNOWFLAKE_DATABASE", "PRODDB"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
        role=os.environ.get("SNOWFLAKE_ROLE", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING PARSING & COSINE SIMILARITY — stdlib only
# ─────────────────────────────────────────────────────────────────────────────

def parse_embedding(raw) -> Optional[List[float]]:
    """
    Parse EMBEDDING_256 from a Snowflake VARIANT column.
    The connector may return it as a Python list already, or as a JSON string.
    Returns None if the value is missing or unparseable.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two equal-length vectors using stdlib math.
    Returns float in [-1.0, 1.0]; 1.0 = identical direction.
    Returns 0.0 if either vector is zero-length.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────────────────────
# UNION-FIND  (path-compressed, for O(α·n²) clustering)
# ─────────────────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

    def clusters(self, n: int) -> List[List[int]]:
        groups: Dict[int, List[int]] = collections.defaultdict(list)
        for i in range(n):
            groups[self.find(i)].append(i)
        return list(groups.values())


# ─────────────────────────────────────────────────────────────────────────────
# CORE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# Each item is (title, embedding_vector).
Item = Tuple[str, List[float]]


def find_title_groups(items: List[Item], threshold: float) -> List[List[str]]:
    """
    Cluster titles by pairwise cosine similarity of their EMBEDDING_256 vectors.
    Items whose embedding is None are each placed in their own singleton cluster.
    Returns list of clusters (each cluster is a list of title strings).
    """
    n = len(items)
    uf = UnionFind(n)
    for i in range(n):
        emb_i = items[i][1]
        if emb_i is None:
            continue
        for j in range(i + 1, n):
            emb_j = items[j][1]
            if emb_j is None:
                continue
            if cosine_similarity(emb_i, emb_j) >= threshold:
                uf.union(i, j)
    return [[items[idx][0] for idx in group] for group in uf.clusters(n)]


def analyze(rows: List[Tuple]) -> List[Dict]:
    """
    rows: iterable of (consumer_id, daypart, title, embedding_256_raw)
    Returns list of per-pair dicts sorted by max_cluster_size desc.
    """
    # Group (title, embedding) pairs by (consumer_id, daypart)
    grouped: Dict[Tuple, List[Item]] = collections.defaultdict(list)
    skipped_no_emb = 0
    for consumer_id, daypart, title, emb_raw in rows:
        if not title:
            continue
        emb = parse_embedding(emb_raw)
        if emb is None:
            skipped_no_emb += 1
        grouped[(consumer_id, daypart)].append((title, emb))

    if skipped_no_emb:
        print(f"  Warning: {skipped_no_emb} rows had missing/unparseable embeddings "
              f"(treated as singletons).")

    results = []
    for (consumer_id, daypart), items in grouped.items():
        clusters = find_title_groups(items, SIMILARITY_THRESHOLD)
        clusters.sort(key=len, reverse=True)
        max_sz = len(clusters[0]) if clusters else 0
        results.append({
            "consumer_id": consumer_id,
            "daypart": daypart,
            "n_titles": len(items),
            "n_groups": len(clusters),
            "max_cluster_size": max_sz,
            "clusters": clusters,
        })

    results.sort(key=lambda r: -r["max_cluster_size"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: List[Dict]) -> None:
    total_pairs = len(results)
    sep = "═" * 72

    print(f"\n{sep}")
    print("  CAROUSEL TITLE SIMILARITY ANALYSIS")
    print(f"  Table  : {TABLE}")
    print(f"  Similarity: cosine(EMBEDDING_256) >= {SIMILARITY_THRESHOLD}  |  Consumers sampled: {SAMPLE_CONSUMERS}")
    print(f"  Total <consumer_id, daypart> pairs analysed: {total_pairs}")
    print(sep)

    # ── 1. Distribution of n_groups ──────────────────────────────────────────
    n_groups_counter = collections.Counter(r["n_groups"] for r in results)
    avg_groups = sum(r["n_groups"] for r in results) / total_pairs
    avg_titles = sum(r["n_titles"] for r in results) / total_pairs

    print("\n  Distribution of 'number of distinct title groups' per <consumer_id, daypart>")
    print(f"  (a group = cluster of similar titles; singletons count as groups of 1)\n")
    print(f"  {'N groups':<10} {'# pairs':>8}  {'% of pairs':>11}")
    print(f"  {'-'*10} {'-'*8}  {'-'*11}")
    for n_groups in sorted(n_groups_counter):
        count = n_groups_counter[n_groups]
        pct = 100.0 * count / total_pairs
        bar = "█" * max(1, round(pct / 2))
        print(f"  {n_groups:<10} {count:>8}  {pct:>10.1f}%  {bar}")

    print(f"\n  Average titles per pair : {avg_titles:.1f}")
    print(f"  Average groups per pair : {avg_groups:.2f}")

    # ── 2. Pairs with large similar-title clusters ────────────────────────────
    interesting = [r for r in results if r["max_cluster_size"] > CLUSTER_SIZE_THRESHOLD]

    print(f"\n{sep}")
    print(
        f"  PAIRS WHERE A SIMILAR-TITLE CLUSTER HAS SIZE > {CLUSTER_SIZE_THRESHOLD}"
        f"  ({len(interesting)} pairs)"
    )
    print(sep)

    if not interesting:
        print(f"  None found at cosine threshold={SIMILARITY_THRESHOLD}.")
        return

    for r in interesting[:MAX_EXAMPLES]:
        print(f"\n  Consumer: {r['consumer_id']}  |  Daypart: {r['daypart']}")
        print(f"  Total titles: {r['n_titles']}  →  Distinct groups: {r['n_groups']}")
        for cluster in r["clusters"]:
            if len(cluster) > CLUSTER_SIZE_THRESHOLD:
                titles_str = "\n      ".join(f'"{t}"' for t in cluster)
                print(f"    Cluster size {len(cluster)}:")
                print(f"      {titles_str}")

    if len(interesting) > MAX_EXAMPLES:
        print(f"\n  ... and {len(interesting) - MAX_EXAMPLES} more pairs (truncated).")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Connecting to Snowflake...")
    conn = get_connection()
    cursor = conn.cursor()

    if SAMPLE_CONSUMERS:
        query = f"""
            WITH sampled AS (
                SELECT DISTINCT CONSUMER_ID
                FROM {TABLE}
                LIMIT {SAMPLE_CONSUMERS}
            )
            SELECT t.CONSUMER_ID, t.DAYPART, t.TITLE, t.EMBEDDING_256
            FROM {TABLE} t
            JOIN sampled s ON t.CONSUMER_ID = s.CONSUMER_ID
            WHERE t.TITLE IS NOT NULL
        """
    else:
        query = f"""
            SELECT CONSUMER_ID, DAYPART, TITLE, EMBEDDING_256
            FROM {TABLE}
            WHERE TITLE IS NOT NULL
        """

    print(f"Fetching titles + EMBEDDING_256 for up to {SAMPLE_CONSUMERS} consumers...")
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"  {len(rows):,} rows fetched.")

    print("Clustering similar titles by cosine similarity per <consumer_id, daypart>...")
    results = analyze(rows)
    print_report(results)


if __name__ == "__main__":
    main()
