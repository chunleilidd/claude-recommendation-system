#!/usr/bin/env python3
"""
Compare V3 and V4 customer profiles across five field pairs:
  1. cuisine_preferences        — filtered cuisine names → set IOU
                                  V4 filter: pct > 10% OR pct × total_orders >= 5
  2. food_preferences (V3)
     vs dish_preferences (V4)  — semantic soft-IOU via sentence embeddings
  3. taste_preference (V3)
     vs dish_preferences (V4)  — semantic soft-IOU via sentence embeddings
  4. top_ordered_items          — word 1-gram, 2-gram, 3-gram IOU + coverage (per daypart)
  5. top_ordered_store_ids (V3)
     vs top_ordered_stores_ids (V4) — exact ID IOU (per daypart)

Semantic soft-IOU (BERTScore-style):
  Encode each V3 phrase and V4 dish item with paraphrase-MiniLM-L6-v2.
  precision = mean(max cosine-sim of each V4 item to any V3 phrase)
  recall    = mean(max cosine-sim of each V3 phrase to any V4 item)
  soft-IOU  = F1 = 2 * P * R / (P + R)
"""

import csv
import json
import re
from collections import defaultdict
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

CSV_FILE = (
    '/Users/chunlei.li/Documents/claude_recommendation_system/'
    'cx_profile_v3_v4_comparison - _RFC__Cx_Profile_4_0 (4).csv'
)

DAYPARTS = [
    'weekday_breakfast', 'weekday_lunch', 'weekday_dinner', 'weekday_late_night',
    'weekend_breakfast', 'weekend_lunch', 'weekend_dinner', 'weekend_late_night',
]

MIN_ORDERS = 5  # minimum daypart orders to include in daypart-level comparisons

KNOWN_CUISINES = frozenset({
    # True cuisine / food-culture categories only
    # (dish-type terms like pizza, sandwiches, deli, burgers removed to avoid
    #  false matches in style descriptors like "deli-style" or narrative tails)
    'american', 'italian', 'indian', 'chinese', 'mexican', 'sichuan', 'tex-mex',
    'japanese', 'thai', 'greek', 'mediterranean', 'korean', 'vietnamese', 'french',
    'caribbean', 'cajun', 'southern', 'bbq', 'barbecue', 'seafood', 'asian',
    'latin', 'spanish', 'ethiopian', 'peruvian', 'north indian', 'south indian',
    'northern indian', 'desi', 'bangladeshi', 'halal', 'kosher', 'turkish',
    'lebanese', 'persian', 'afghani', 'pakistani', 'taiwanese', 'cantonese',
    'szechuan', 'mongolian', 'indonesian', 'filipino', 'singaporean',
    'middle eastern', 'fusion', 'hawaiian', 'sichuan chinese', 'new american',
    'tibetan', 'nepalese', 'dim sum',
})

STOP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'with', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
    'should', 'may', 'might', 'must', 'can', 'could', 'that', 'this',
    'these', 'those', 'it', 'its', 'from', 'by', 'as', 'very', 'more',
    'most', 'some', 'any', 'all', 'both', 'each', 'few', 'other', 'such',
    'no', 'not', 'only', 'same', 'so', 'than', 'too', 'also', 'into',
    'through', 'between', 'about', 'during', 'while', 'although', 'because',
    'however', 'their', 'they', 'them', 'he', 'she', 'his', 'her', 'our',
    'your', 'my', 'we', 'us', 'i', 'you', 'what', 'which', 'who', 'how',
    'when', 'where', 'why', 'there', 'here', 'up', 'out', 'if', 'then',
    'just', 'well', 'even', 'back', 'still', 'old', 'often', 'usually',
    'always', 'never', 'sometimes', 'primarily', 'strong', 'emphasis', 'focus',
    'consistent', 'occasional', 'frequent', 'given', 'across', 'pattern',
    'overall', 'general', 'mix', 'wide', 'variety', 'range', 'include',
    'including', 'particularly', 'especially', 'typically', 'mainly', 'mostly',
    'largely', 'relatively', 'fairly', 'quite', 'rather', 'somewhat', 'highly',
    'based', 'reflects', 'indicate', 'shows', 'likely', 'predominantly',
    'preference', 'preferences', 'orders', 'order', 'flavors', 'dishes',
    'foods', 'items', 'meals', 'cuisine', 'cuisines', 'style', 'type', 'types',
    'common', 'like', 'clear', 'regular', 'new',
})


CUISINE_ALIASES = {
    'northern indian':  'north indian',
    'barbecue':         'bbq',
    'szechuan':         'sichuan',
    'sichuan chinese':  'sichuan',
    'tex mex':          'tex-mex',
}


def normalize_cuisine(name: str) -> str:
    return CUISINE_ALIASES.get(name, name)


# ── Core helpers ──────────────────────────────────────────────────────────────

def set_iou(a: set, b: set):
    """IOU: None if both empty, float otherwise."""
    if not a and not b:
        return None
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def set_coverage(a: set, b: set):
    """Coverage of V4 on V3: |V3 ∩ V4| / |V3|. None if V3 is empty."""
    if not a:
        return None
    return len(a & b) / len(a)


def daypart_order_count(v4_dp_dict: dict) -> int:
    """Parse order count from V4 cuisine_preferences prefix 'N orders: ...'"""
    raw = v4_dp_dict.get('cuisine_preferences', '')
    m = re.match(r'^(\d+)\s+orders', raw)
    return int(m.group(1)) if m else 0


def word_ngrams_from_items(items: list, n: int) -> set:
    """Word n-grams as frozensets from a list of dish/item name strings.

    Each item is tokenised independently (stop words removed), then all
    size-n combinations of its tokens are added as frozensets.  Using
    frozensets (unordered) instead of tuples (ordered) means
    frozenset({'chicken','salad'}) matches both 'chicken salad' and
    'salad chicken', cleanly handling permutation equivalence without
    enumerating every ordering.  Words from different items never mix.

    n=1 → bag-of-words (frozensets of one word each)
    n=2 → unordered word pairs
    n=3 → unordered word triples
    """
    result = set()
    for item in items:
        tokens = [w for w in re.findall(r'\b[a-z]+\b', item.lower())
                  if w not in STOP_WORDS and len(w) >= 2]
        for combo in combinations(tokens, n):
            result.add(frozenset(combo))
    return result


def normalize_items(items: list) -> set:
    """Represent each item as a frozenset of its words for whole-item matching.

    'chicken salad' and 'salad chicken' both become frozenset({'chicken','salad'})
    so they match regardless of word order.  Items with no valid tokens are dropped.
    """
    result = set()
    for item in items:
        tokens = frozenset(w for w in re.findall(r'\b[a-z]+\b', item.lower())
                           if w not in STOP_WORDS and len(w) >= 2)
        if tokens:
            result.add(tokens)
    return result


# ── Cuisine extraction ────────────────────────────────────────────────────────

def extract_cuisines_v3(text: str) -> set:
    """Scan V3 free text for known cuisine terms (longest match first).
    Only terms present in KNOWN_CUISINES are returned.
    Restricted to the portion before the first semicolon/colon/dash to avoid
    matching cuisine-like words in descriptive narrative tails."""
    if not text:
        return set()
    text_head = re.split(r'[;:–—]', text)[0]
    found = set()
    remaining = text_head.lower()
    for cuisine in sorted(KNOWN_CUISINES, key=len, reverse=True):
        if cuisine in remaining:
            found.add(cuisine)
            remaining = remaining.replace(cuisine, ' ')
    return {normalize_cuisine(c) for c in found}


def extract_cuisines_v4_filtered(text: str,
                                  pct_threshold: float = 10.0,
                                  count_threshold: float = 5.0) -> set:
    """
    Parse V4 cuisine format:  "N orders: X% cuisine1, Y% cuisine2, cuisine3, ..."
    Each %-token marks a new entry (may be a combined tag like 'Italian, American').
    Include entry if: percentage > pct_threshold OR pct × total_orders / 100 >= count_threshold
    """
    if not text:
        return set()
    m = re.match(r'^(\d+)\s+orders:\s*', text)
    total_orders = int(m.group(1)) if m else 0
    body = text[m.end():] if m else text

    markers = list(re.finditer(r'(\d+\.?\d*)%\s*', body))
    result = set()
    for i, marker in enumerate(markers):
        pct = float(marker.group(1))
        start = marker.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(body)
        tag = body[start:end].strip().rstrip(', ')

        order_count = pct * total_orders / 100.0
        if pct > pct_threshold or order_count >= count_threshold:
            for part in tag.split(','):
                part = part.strip().lower().strip('.,;-–—')
                if part and len(part) >= 2:
                    result.add(normalize_cuisine(part))
    return result


def extract_cuisines_v4_top5(text: str, top_n: int = 5) -> set:
    """Take the top-N percentage entries from V4 cuisine_preferences."""
    if not text:
        return set()
    m = re.match(r'^(\d+)\s+orders:\s*', text)
    body = text[m.end():] if m else text
    markers = list(re.finditer(r'(\d+\.?\d*)%\s*', body))
    result = set()
    for i, marker in enumerate(markers[:top_n]):
        start = marker.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(body)
        tag = body[start:end].strip().rstrip(', ')
        for part in tag.split(','):
            part = part.strip().lower().strip('.,;-–—')
            if part and len(part) >= 2:
                result.add(normalize_cuisine(part))
    return result


# ── Text → phrase splitter for semantic comparison ───────────────────────────

def text_to_phrases(text: str, max_phrases: int = 20) -> list:
    """
    Split a free-text description into meaningful phrases for embedding.
    Splits on semicolons and periods first, then commas within long segments.
    """
    if not text:
        return []
    phrases = []
    for part in re.split(r'[.;]\s*', text):
        sub = part.split(',')
        if len(sub) > 1:
            for sp in sub:
                sp = re.sub(r'\([^)]*\)', '', sp).strip()
                if len(sp) >= 4:
                    phrases.append(sp)
        else:
            part = re.sub(r'\([^)]*\)', '', part).strip()
            if len(part) >= 4:
                phrases.append(part)
    return phrases[:max_phrases]


# ── Field accessors ───────────────────────────────────────────────────────────

def v3_overall(v3): return v3.get('overall_profile') or {}
def v4_overall(v4): return v4.get('overall') or {}
def v3_dp(v3, dp): return ((v3.get('breakdown') or {}).get('daypart') or {}).get(dp) or {}
def v4_dp(v4, dp): return (v4.get('breakdown') or {}).get(dp) or {}


def parse_dish_prefs(raw) -> list:
    if not raw:
        return []
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass
    return []


def parse_v3_top_items(v3, dp) -> list:
    raw = (v3.get('breakdown') or {}).get('top_ordered_items', {}).get(dp, '')
    return [t.strip() for t in raw.split(',') if t.strip()]


def parse_v4_top_items(dp_dict) -> list:
    raw = dp_dict.get('top_ordered_items', '')
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            items = []
            for v in obj.values():
                items.extend(t.strip() for t in str(v).split(',') if t.strip())
            return items
    except json.JSONDecodeError:
        pass
    return [t.strip() for t in raw.split(',') if t.strip()]


def parse_v3_store_ids(v3, dp) -> set:
    raw = (v3.get('breakdown') or {}).get('top_ordered_store_ids', {}).get(dp, '')
    return {t.strip() for t in str(raw).split(',') if t.strip()}


def parse_v4_store_ids(dp_dict) -> set:
    raw = dp_dict.get('top_ordered_stores_ids', '')
    if not raw:
        return set()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            ids = set()
            for v in obj.values():
                cleaned = str(v).strip().strip('[]')
                ids.update(t.strip() for t in cleaned.split(',') if t.strip())
            return ids
    except json.JSONDecodeError:
        pass
    return set()


# ── Load data ─────────────────────────────────────────────────────────────────

with open(CSV_FILE, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} consumer profiles.")

# ── Phase 1: collect all unique texts for batch embedding ─────────────────────
# We embed V3 food/taste phrases and V4 dish items all in one shot.

print("Collecting texts for embedding...")
text_registry: set[str] = set()

# Store per-row extracted data to avoid re-parsing in the main loop
row_cache = []

for row in rows:
    try:
        v3 = json.loads(row['profile_v3'])
        v4 = json.loads(row['profile_v4'])
    except Exception:
        row_cache.append(None)
        continue

    cache = {
        'v3': v3, 'v4': v4,
        'fp3_ov':   text_to_phrases(v3_overall(v3).get('food_preferences', '')),
        'tp3_ov':   [v3_overall(v3).get('taste_preference', '')] if v3_overall(v3).get('taste_preference') else [],
        'dish4_ov': parse_dish_prefs(v4_overall(v4).get('dish_preferences', '')),
        'dp': {}
    }
    for t in cache['fp3_ov'] + cache['tp3_ov'] + cache['dish4_ov']:
        text_registry.add(t)

    for dp in DAYPARTS:
        d3 = v3_dp(v3, dp)
        d4 = v4_dp(v4, dp)
        fp3d    = text_to_phrases(d3.get('food_preferences', ''))
        tp3d    = [d3.get('taste_preference', '')] if d3.get('taste_preference') else []
        dish4d  = parse_dish_prefs(d4.get('dish_preferences', ''))
        items3d = parse_v3_top_items(v3, dp)
        items4d = parse_v4_top_items(d4)
        cache['dp'][dp] = {
            'fp3': fp3d, 'tp3': tp3d, 'dish4': dish4d,
            'items3': items3d, 'items4': items4d,
        }
        for t in fp3d + tp3d + dish4d + items3d + items4d:
            text_registry.add(t)

    row_cache.append(cache)

# ── Phase 2: batch encode all texts ──────────────────────────────────────────

text_list = [t for t in text_registry if t]   # remove blanks
print(f"Encoding {len(text_list)} unique phrases with paraphrase-MiniLM-L6-v2 ...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
all_embs = model.encode(text_list, normalize_embeddings=True, batch_size=128, show_progress_bar=True).astype(np.float64)
emb_map = {t: all_embs[i] for i, t in enumerate(text_list)}


def semantic_soft_iou(v3_phrases: list, v4_items: list):
    """
    BERTScore-style soft IOU using pre-computed sentence embeddings.
      precision = mean(for each V4 item: max cosine-sim to any V3 phrase)
      recall    = mean(for each V3 phrase: max cosine-sim to any V4 item)
      soft-IOU  = harmonic mean (F1)
    Returns (soft_iou, recall) tuple, or None if either side is empty.
    recall serves as semantic coverage: how well V4 covers each V3 phrase.
    """
    v3_embs = np.array([emb_map[t] for t in v3_phrases if t in emb_map])
    v4_embs = np.array([emb_map[t] for t in v4_items   if t in emb_map])
    if v3_embs.size == 0 or v4_embs.size == 0:
        return None
    sim = np.nan_to_num(v3_embs @ v4_embs.T, nan=0.0)  # [|v3| x |v4|]
    precision = float(sim.max(axis=0).mean())
    recall    = float(sim.max(axis=1).mean())
    if precision + recall < 1e-9:
        return (0.0, recall)
    soft_iou = 2 * precision * recall / (precision + recall)
    return (soft_iou, recall)


# ── Phase 3: main comparison loop ─────────────────────────────────────────────

cuisine_iou_ov     = [];  cuisine_cov_ov     = []
cuisine_iou_dp     = defaultdict(list);  cuisine_cov_dp     = defaultdict(list)
cuisine_t5_iou_ov  = [];  cuisine_t5_cov_ov  = []
cuisine_t5_iou_dp  = defaultdict(list);  cuisine_t5_cov_dp  = defaultdict(list)
food_dish_iou_ov   = [];  food_dish_cov_ov   = []
food_dish_iou_dp   = defaultdict(list);  food_dish_cov_dp   = defaultdict(list)
taste_dish_iou_ov  = [];  taste_dish_cov_ov  = []
taste_dish_iou_dp  = defaultdict(list);  taste_dish_cov_dp  = defaultdict(list)
items_1g_iou_dp  = defaultdict(list);  items_1g_cov_dp  = defaultdict(list)
items_2g_iou_dp  = defaultdict(list);  items_2g_cov_dp  = defaultdict(list)
items_3g_iou_dp  = defaultdict(list);  items_3g_cov_dp  = defaultdict(list)
items_sem_iou_dp = defaultdict(list);  items_sem_cov_dp = defaultdict(list)
stores_iou_dp      = defaultdict(list);  stores_cov_dp      = defaultdict(list)

for idx, row in enumerate(rows):
    cache = row_cache[idx]
    if cache is None:
        continue
    v3, v4 = cache['v3'], cache['v4']

    # ── 1. cuisine_preferences overall ────────────────────────────────────────
    c3     = extract_cuisines_v3(v3_overall(v3).get('cuisine_preferences', ''))
    c4     = extract_cuisines_v4_filtered(v4_overall(v4).get('cuisine_preferences', ''))
    c4_t5  = extract_cuisines_v4_top5(v4_overall(v4).get('cuisine_preferences', ''))
    if set_iou(c3, c4)      is not None: cuisine_iou_ov.append(set_iou(c3, c4))
    if set_coverage(c3, c4) is not None: cuisine_cov_ov.append(set_coverage(c3, c4))
    if set_iou(c3, c4_t5)      is not None: cuisine_t5_iou_ov.append(set_iou(c3, c4_t5))
    if set_coverage(c3, c4_t5) is not None: cuisine_t5_cov_ov.append(set_coverage(c3, c4_t5))

    # ── 2 & 3. food / taste vs dish — overall (semantic) ─────────────────────
    result = semantic_soft_iou(cache['fp3_ov'], cache['dish4_ov'])
    if result is not None:
        food_dish_iou_ov.append(result[0]); food_dish_cov_ov.append(result[1])

    result = semantic_soft_iou(cache['tp3_ov'], cache['dish4_ov'])
    if result is not None:
        taste_dish_iou_ov.append(result[0]); taste_dish_cov_ov.append(result[1])

    # ── Per-daypart ────────────────────────────────────────────────────────────
    for dp in DAYPARTS:
        d3 = v3_dp(v3, dp)
        d4 = v4_dp(v4, dp)
        dp_cache = cache['dp'][dp]

        # 1. cuisine per daypart — only if >= MIN_ORDERS
        if daypart_order_count(d4) >= MIN_ORDERS:
            c3d    = extract_cuisines_v3(d3.get('cuisine_preferences', ''))
            c4d    = extract_cuisines_v4_filtered(d4.get('cuisine_preferences', ''))
            c4d_t5 = extract_cuisines_v4_top5(d4.get('cuisine_preferences', ''))
            if set_iou(c3d, c4d)       is not None: cuisine_iou_dp[dp].append(set_iou(c3d, c4d))
            if set_coverage(c3d, c4d)  is not None: cuisine_cov_dp[dp].append(set_coverage(c3d, c4d))
            if set_iou(c3d, c4d_t5)       is not None: cuisine_t5_iou_dp[dp].append(set_iou(c3d, c4d_t5))
            if set_coverage(c3d, c4d_t5)  is not None: cuisine_t5_cov_dp[dp].append(set_coverage(c3d, c4d_t5))

        # 2. food_preferences vs dish_preferences (semantic)
        result = semantic_soft_iou(dp_cache['fp3'], dp_cache['dish4'])
        if result is not None:
            food_dish_iou_dp[dp].append(result[0]); food_dish_cov_dp[dp].append(result[1])

        # 3. taste_preference vs dish_preferences (semantic)
        result = semantic_soft_iou(dp_cache['tp3'], dp_cache['dish4'])
        if result is not None:
            taste_dish_iou_dp[dp].append(result[0]); taste_dish_cov_dp[dp].append(result[1])

        # 4. top_ordered_items — word 1/2/3-gram + semantic IOU + coverage
        items3 = dp_cache['items3']
        items4 = dp_cache['items4']
        if items3 or items4:
            for n, iou_dp, cov_dp in [
                (1, items_1g_iou_dp, items_1g_cov_dp),
                (2, items_2g_iou_dp, items_2g_cov_dp),
                (3, items_3g_iou_dp, items_3g_cov_dp),
            ]:
                g3 = word_ngrams_from_items(items3, n)
                g4 = word_ngrams_from_items(items4, n)
                if set_iou(g3, g4)      is not None: iou_dp[dp].append(set_iou(g3, g4))
                if set_coverage(g3, g4) is not None: cov_dp[dp].append(set_coverage(g3, g4))

            result = semantic_soft_iou(items3, items4)
            if result is not None:
                items_sem_iou_dp[dp].append(result[0])
                items_sem_cov_dp[dp].append(result[1])

        # 5. store IDs — exact IOU + coverage
        s3 = parse_v3_store_ids(v3, dp)
        s4 = parse_v4_store_ids(d4)
        if set_iou(s3, s4)      is not None: stores_iou_dp[dp].append(set_iou(s3, s4))
        if set_coverage(s3, s4) is not None: stores_cov_dp[dp].append(set_coverage(s3, s4))


# ── Output helpers ─────────────────────────────────────────────────────────────

MD_FILE = (
    '/Users/chunlei.li/Documents/claude_recommendation_system/'
    'profile_v3_v4_comparison_results.md'
)

def mean(vals):
    return sum(vals) / len(vals) if vals else None

def fv(vals):
    """Plain float string or —."""
    m = mean(vals)
    return f"{m:.3f}" if m is not None else "—"

def fn(vals):
    return str(len(vals)) if vals else "—"


def build_simple_table(title, note, ov_iou, dp_iou, ov_cov, dp_cov):
    """Return (console_str, md_str) for a 2-metric table (IOU + Coverage)."""
    levels = []
    if ov_iou or ov_cov:
        levels.append(('OVERALL', ov_iou, ov_cov))
    all_iou, all_cov = [], []
    for dp in DAYPARTS:
        iv, cv = dp_iou.get(dp, []), dp_cov.get(dp, [])
        if iv or cv:
            levels.append(('  ' + dp, iv, cv))
            all_iou.extend(iv); all_cov.extend(cv)
    if all_iou or all_cov:
        levels.append(('  ALL DAYPARTS', all_iou, all_cov))

    # console
    W = 26
    lines = [f"\n{'─'*70}", title]
    if note: lines.append(f"  {note}")
    lines.append(f"{'─'*70}")
    lines.append(f"  {'Level':<{W}}  {'IOU':>7}  {'n':>4}  {'Coverage':>8}  {'n':>4}")
    lines.append(f"  {'-'*W}  {'-'*7}  {'-'*4}  {'-'*8}  {'-'*4}")
    for lbl, iv, cv in levels:
        lines.append(f"  {lbl:<{W}}  {fv(iv):>7}  {fn(iv):>4}  {fv(cv):>8}  {fn(cv):>4}")
    con = '\n'.join(lines)

    # markdown
    md = [f"### {title}", f"*{note}*" if note else '',
          '| Level | IOU | n | Coverage | n |',
          '|---|---:|---:|---:|---:|']
    for lbl, iv, cv in levels:
        md.append(f"| {lbl.strip()} | {fv(iv)} | {fn(iv)} | {fv(cv)} | {fn(cv)} |")
    return con, '\n'.join(md)


def build_cuisine_table(title, note,
                        ov_iou, dp_iou, ov_cov, dp_cov,
                        ov_t5_iou, dp_t5_iou, ov_t5_cov, dp_t5_cov):
    """4-metric cuisine table: threshold IOU/Cov + top-5 IOU/Cov."""
    levels = []
    if ov_iou or ov_cov:
        levels.append(('OVERALL', ov_iou, ov_cov, ov_t5_iou, ov_t5_cov))
    all_i, all_c, all_ti, all_tc = [], [], [], []
    for dp in DAYPARTS:
        iv, cv   = dp_iou.get(dp, []),    dp_cov.get(dp, [])
        tiv, tcv = dp_t5_iou.get(dp, []), dp_t5_cov.get(dp, [])
        if iv or cv:
            levels.append(('  ' + dp, iv, cv, tiv, tcv))
            all_i.extend(iv); all_c.extend(cv)
            all_ti.extend(tiv); all_tc.extend(tcv)
    if all_i or all_c:
        levels.append(('  ALL DAYPARTS', all_i, all_c, all_ti, all_tc))

    W = 26
    lines = [f"\n{'─'*90}", title]
    if note: lines.append(f"  {note}")
    lines.append(f"{'─'*90}")
    lines.append(f"  {'Level':<{W}}  {'── Threshold ──':^22}  {'──── Top-5 ────':^22}")
    lines.append(f"  {'':^{W}}  {'IOU':>7}  {'n':>4}  {'Cov':>7}  {'n':>4}  {'IOU':>7}  {'n':>4}  {'Cov':>7}  {'n':>4}")
    lines.append(f"  {'-'*W}  {'-'*7}  {'-'*4}  {'-'*7}  {'-'*4}  {'-'*7}  {'-'*4}  {'-'*7}  {'-'*4}")
    for lbl, iv, cv, tiv, tcv in levels:
        lines.append(
            f"  {lbl:<{W}}  {fv(iv):>7}  {fn(iv):>4}  {fv(cv):>7}  {fn(cv):>4}"
            f"  {fv(tiv):>7}  {fn(tiv):>4}  {fv(tcv):>7}  {fn(tcv):>4}"
        )
    con = '\n'.join(lines)

    md = [f"### {title}", f"*{note}*" if note else '',
          '| Level | Thr IOU | n | Thr Cov | n | Top-5 IOU | n | Top-5 Cov | n |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for lbl, iv, cv, tiv, tcv in levels:
        md.append(f"| {lbl.strip()} | {fv(iv)} | {fn(iv)} | {fv(cv)} | {fn(cv)}"
                  f" | {fv(tiv)} | {fn(tiv)} | {fv(tcv)} | {fn(tcv)} |")
    return con, '\n'.join(md)


ITEM_WEIGHTS = {'1g': 0.10, '2g': 0.20, '3g': 0.30, 'sem': 0.40}


def weighted_item_score(vals_by_label: dict):
    """Weighted combination of 1g/2g/3g/item means.

    Weights 1:2:3:4 (normalized to 0.10/0.20/0.30/0.40).
    Components with no data are excluded and weights are renormalized.
    """
    total_w = 0.0
    weighted_sum = 0.0
    for lbl, w in ITEM_WEIGHTS.items():
        m = mean(vals_by_label.get(lbl, []))
        if m is not None:
            weighted_sum += w * m
            total_w += w
    if total_w == 0:
        return None
    return weighted_sum / total_w


def build_items_table():
    """Word 1/2/3-gram + semantic + weighted IOU/Coverage table for top_ordered_items."""
    cols = [
        ('1g',  items_1g_iou_dp,  items_1g_cov_dp),
        ('2g',  items_2g_iou_dp,  items_2g_cov_dp),
        ('3g',  items_3g_iou_dp,  items_3g_cov_dp),
        ('sem', items_sem_iou_dp, items_sem_cov_dp),
    ]
    W = 26
    lines = [
        f"\n{'─'*130}",
        "4. top_ordered_items  [word n-grams + semantic, per daypart]",
        f"  Weights: 1g=0.10  2g=0.20  3g=0.30  sem=0.40",
        f"{'─'*130}",
        f"  {'Level':<{W}}  {'1g IOU':>7}  {'n':>4}  {'1g Cov':>7}  {'n':>4}"
        f"  {'2g IOU':>7}  {'n':>4}  {'2g Cov':>7}  {'n':>4}"
        f"  {'3g IOU':>7}  {'n':>4}  {'3g Cov':>7}  {'n':>4}"
        f"  {'Sem IOU':>7}  {'n':>4}  {'Sem Cov':>7}  {'n':>4}"
        f"  {'W IOU':>7}  {'W Cov':>7}",
        f"  {'-'*W}  {'-'*7}  {'-'*4}  {'-'*7}  {'-'*4}"
        f"  {'-'*7}  {'-'*4}  {'-'*7}  {'-'*4}"
        f"  {'-'*7}  {'-'*4}  {'-'*7}  {'-'*4}"
        f"  {'-'*7}  {'-'*4}  {'-'*7}  {'-'*4}"
        f"  {'-'*7}  {'-'*7}",
    ]
    md = [
        "### 4. top_ordered_items  [word n-grams + semantic, per daypart]",
        "*Weights: 1g=0.10, 2g=0.20, 3g=0.30, sem=0.40 (harder match → higher weight)*",
        "| Level | 1g IOU | n | 1g Cov | n | 2g IOU | n | 2g Cov | n | 3g IOU | n | 3g Cov | n | Sem IOU | n | Sem Cov | n | W IOU | W Cov |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    totals = {lbl: ([], []) for lbl, _, _ in cols}
    for dp in DAYPARTS:
        row_vals = [(lbl, iou_dp.get(dp, []), cov_dp.get(dp, []))
                    for lbl, iou_dp, cov_dp in cols]
        if not row_vals[0][1]: continue
        for lbl, iv, cv in row_vals:
            totals[lbl][0].extend(iv); totals[lbl][1].extend(cv)
        w_iou = weighted_item_score({lbl: iv for lbl, iv, _ in row_vals})
        w_cov = weighted_item_score({lbl: cv for lbl, _, cv in row_vals})
        w_iou_s = f"{w_iou:.3f}" if w_iou is not None else "—"
        w_cov_s = f"{w_cov:.3f}" if w_cov is not None else "—"
        cell = ''.join(f"  {fv(iv):>7}  {fn(iv):>4}  {fv(cv):>7}  {fn(cv):>4}"
                       for _, iv, cv in row_vals)
        lines.append(f"  {'  '+dp:<{W}}{cell}  {w_iou_s:>7}  {w_cov_s:>7}")
        md.append(f"| {dp} |" + '|'.join(f" {fv(iv)} | {fn(iv)} | {fv(cv)} | {fn(cv)} "
                                          for _, iv, cv in row_vals)
                  + f'| {w_iou_s} | {w_cov_s} |')
    if totals['1g'][0]:
        cell = ''.join(f"  {fv(totals[lbl][0]):>7}  {fn(totals[lbl][0]):>4}"
                       f"  {fv(totals[lbl][1]):>7}  {fn(totals[lbl][1]):>4}"
                       for lbl, _, _ in cols)
        w_iou = weighted_item_score({lbl: totals[lbl][0] for lbl, _, _ in cols})
        w_cov = weighted_item_score({lbl: totals[lbl][1] for lbl, _, _ in cols})
        w_iou_s = f"{w_iou:.3f}" if w_iou is not None else "—"
        w_cov_s = f"{w_cov:.3f}" if w_cov is not None else "—"
        lines.append(f"  {'  ALL DAYPARTS':<{W}}{cell}  {w_iou_s:>7}  {w_cov_s:>7}")
        md.append(f"| ALL DAYPARTS |" + '|'.join(
            f" {fv(totals[lbl][0])} | {fn(totals[lbl][0])} | {fv(totals[lbl][1])} | {fn(totals[lbl][1])} "
            for lbl, _, _ in cols) + f'| {w_iou_s} | {w_cov_s} |')
    return '\n'.join(lines), '\n'.join(md)


# ── Build all sections ─────────────────────────────────────────────────────────

con1, md1 = build_cuisine_table(
    "1. cuisine_preferences  [exact set]",
    "V3: KNOWN_CUISINES scan | V4 threshold: pct>10% OR count>=5 | V4 top-5: top 5 pct entries | +aliases",
    cuisine_iou_ov, cuisine_iou_dp, cuisine_cov_ov, cuisine_cov_dp,
    cuisine_t5_iou_ov, cuisine_t5_iou_dp, cuisine_t5_cov_ov, cuisine_t5_cov_dp,
)

con2, md2 = build_simple_table(
    "2. food_preferences (V3)  vs  dish_preferences (V4)  [semantic]",
    "Embedding: paraphrase-MiniLM-L6-v2 | IOU = soft-IOU (F1) | Coverage = recall",
    food_dish_iou_ov, food_dish_iou_dp, food_dish_cov_ov, food_dish_cov_dp,
)

con3, md3 = build_simple_table(
    "3. taste_preference (V3)  vs  dish_preferences (V4)  [semantic]",
    "Embedding: paraphrase-MiniLM-L6-v2 | IOU = soft-IOU (F1) | Coverage = recall",
    taste_dish_iou_ov, taste_dish_iou_dp, taste_dish_cov_ov, taste_dish_cov_dp,
)

con4, md4 = build_items_table()

con5, md5 = build_simple_table(
    "5. top_ordered_stores  [exact ID set, per daypart]",
    "Coverage = |V3 stores ∩ V4 stores| / |V3 stores|",
    [], stores_iou_dp, [], stores_cov_dp,
)

# ── Print to console ───────────────────────────────────────────────────────────

header = "\n" + "=" * 70 + "\nPROFILE V3 vs V4 — FIELD COMPARISON SUMMARY\n" + "=" * 70
legend = """
Metrics:
  IOU      = |V3 ∩ V4| / |V3 ∪ V4|
  Coverage = |V3 ∩ V4| / |V3|  (fraction of V3 found in V4)
  Semantic IOU      = soft-IOU F1 (BERTScore-style, paraphrase-MiniLM-L6-v2)
  Semantic Coverage = recall  (mean max-sim of each V3 phrase to any V4 item)
  Cuisine daypart: >= 5 orders required
  Cuisine top-5: top 5 percentage-ranked entries from V4
"""
print(header + legend)
for block in [con1, con2, con3, con4, con5]:
    print(block)

# ── Write markdown ─────────────────────────────────────────────────────────────

from datetime import date

md_header = f"""# Profile V3 vs V4 — Field Comparison Summary

**Generated:** {date.today()}
**Dataset:** 52 consumers
**Embedding model:** paraphrase-MiniLM-L6-v2

## Metrics

| Metric | Definition |
|---|---|
| IOU | \\|V3 ∩ V4\\| / \\|V3 ∪ V4\\| |
| Coverage | \\|V3 ∩ V4\\| / \\|V3\\| — fraction of V3 found in V4 |
| Semantic IOU | Soft-IOU F1 (BERTScore-style) |
| Semantic Coverage | Recall — mean max-sim of each V3 phrase to any V4 item |

> Cuisine daypart comparisons require ≥ 5 orders.
> Cuisine **threshold** filter: pct > 10% OR count ≥ 5.
> Cuisine **top-5**: top 5 percentage-ranked entries from V4.

---

"""

with open(MD_FILE, 'w', encoding='utf-8') as f:
    f.write(md_header)
    for block in [md1, md2, md3, md4, md5]:
        f.write(block + '\n\n---\n\n')

print(f"\nResults written to: {MD_FILE}")
