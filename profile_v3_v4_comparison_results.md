# Profile V3 vs V4 — Field Comparison Summary

**Generated:** 2026-03-20
**Dataset:** 52 consumers
**Embedding model:** paraphrase-MiniLM-L6-v2

## Metrics

| Metric | Definition |
|---|---|
| IOU | \|V3 ∩ V4\| / \|V3 ∪ V4\| |
| Coverage | \|V3 ∩ V4\| / \|V3\| — fraction of V3 found in V4 |
| Semantic IOU | Soft-IOU F1 (BERTScore-style) |
| Semantic Coverage | Recall — mean max-sim of each V3 phrase to any V4 item |

> Cuisine daypart comparisons require ≥ 5 orders.
> Cuisine **threshold** filter: pct > 10% OR count ≥ 5.
> Cuisine **top-5**: top 5 percentage-ranked entries from V4.

---

### 1. cuisine_preferences  [exact set]
*V3: KNOWN_CUISINES scan | V4 threshold: pct>10% OR count>=5 | V4 top-5: top 5 pct entries | +aliases*
| Level | Thr IOU | n | Thr Cov | n | Top-5 IOU | n | Top-5 Cov | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OVERALL | 0.427 | 52 | 0.708 | 52 | 0.449 | 52 | 0.631 | 52 |
| weekday_breakfast | 0.458 | 24 | 0.754 | 21 | 0.324 | 24 | 0.905 | 21 |
| weekday_lunch | 0.491 | 51 | 0.756 | 51 | 0.443 | 51 | 0.811 | 51 |
| weekday_dinner | 0.452 | 48 | 0.645 | 48 | 0.500 | 48 | 0.787 | 48 |
| weekday_late_night | 0.341 | 22 | 0.694 | 21 | 0.259 | 22 | 0.734 | 21 |
| weekend_breakfast | 0.468 | 11 | 0.909 | 11 | 0.412 | 11 | 0.955 | 11 |
| weekend_lunch | 0.409 | 27 | 0.658 | 26 | 0.353 | 27 | 0.760 | 26 |
| weekend_dinner | 0.467 | 36 | 0.642 | 36 | 0.470 | 36 | 0.765 | 36 |
| weekend_late_night | 0.368 | 13 | 0.679 | 13 | 0.355 | 13 | 0.769 | 13 |
| ALL DAYPARTS | 0.444 | 232 | 0.700 | 227 | 0.413 | 232 | 0.799 | 227 |

---

### 2. food_preferences (V3)  vs  dish_preferences (V4)  [semantic]
*Embedding: paraphrase-MiniLM-L6-v2 | IOU = soft-IOU (F1) | Coverage = recall*
| Level | IOU | n | Coverage | n |
|---|---:|---:|---:|---:|
| OVERALL | 0.547 | 51 | 0.661 | 51 |
| weekday_breakfast | 0.613 | 39 | 0.702 | 39 |
| weekday_lunch | 0.542 | 51 | 0.671 | 51 |
| weekday_dinner | 0.508 | 50 | 0.603 | 50 |
| weekday_late_night | 0.513 | 39 | 0.581 | 39 |
| weekend_breakfast | 0.609 | 20 | 0.680 | 20 |
| weekend_lunch | 0.509 | 43 | 0.570 | 43 |
| weekend_dinner | 0.506 | 42 | 0.579 | 42 |
| weekend_late_night | 0.465 | 24 | 0.528 | 24 |
| ALL DAYPARTS | 0.531 | 308 | 0.615 | 308 |

---

### 3. taste_preference (V3)  vs  dish_preferences (V4)  [semantic]
*Embedding: paraphrase-MiniLM-L6-v2 | IOU = soft-IOU (F1) | Coverage = recall*
| Level | IOU | n | Coverage | n |
|---|---:|---:|---:|---:|
| OVERALL | 0.383 | 51 | 0.525 | 51 |
| weekday_breakfast | 0.352 | 39 | 0.419 | 39 |
| weekday_lunch | 0.341 | 51 | 0.447 | 51 |
| weekday_dinner | 0.368 | 50 | 0.469 | 50 |
| weekday_late_night | 0.293 | 39 | 0.357 | 39 |
| weekend_breakfast | 0.362 | 20 | 0.433 | 20 |
| weekend_lunch | 0.297 | 43 | 0.350 | 43 |
| weekend_dinner | 0.349 | 42 | 0.433 | 42 |
| weekend_late_night | 0.285 | 24 | 0.340 | 24 |
| ALL DAYPARTS | 0.333 | 308 | 0.411 | 308 |

---

### 4. top_ordered_items  [word n-grams, frozenset/order-invariant, per daypart]
*Weights: 1g=0.10, 2g=0.20, 3g=0.30, item=0.40 (harder match → higher weight)*
| Level | 1g IOU | n | 1g Cov | n | 2g IOU | n | 2g Cov | n | 3g IOU | n | 3g Cov | n | Item IOU | n | Item Cov | n | W IOU | W Cov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weekday_breakfast | 0.371 | 52 | 0.419 | 52 | 0.235 | 44 | 0.265 | 42 | 0.107 | 44 | 0.117 | 42 | 0.005 | 52 | 0.009 | 52 | 0.118 | 0.134 |
| weekday_lunch | 0.399 | 52 | 0.475 | 52 | 0.209 | 52 | 0.236 | 52 | 0.107 | 52 | 0.116 | 52 | 0.034 | 52 | 0.063 | 52 | 0.128 | 0.155 |
| weekday_dinner | 0.393 | 52 | 0.471 | 52 | 0.218 | 52 | 0.255 | 51 | 0.121 | 52 | 0.134 | 51 | 0.037 | 52 | 0.067 | 52 | 0.134 | 0.165 |
| weekday_late_night | 0.356 | 52 | 0.401 | 52 | 0.252 | 42 | 0.291 | 39 | 0.133 | 41 | 0.144 | 39 | 0.029 | 52 | 0.049 | 52 | 0.137 | 0.161 |
| weekend_breakfast | 0.203 | 52 | 0.241 | 52 | 0.216 | 24 | 0.280 | 22 | 0.091 | 23 | 0.118 | 21 | 0.011 | 52 | 0.035 | 52 | 0.095 | 0.129 |
| weekend_lunch | 0.376 | 52 | 0.429 | 52 | 0.215 | 45 | 0.238 | 45 | 0.096 | 45 | 0.103 | 45 | 0.033 | 52 | 0.063 | 52 | 0.123 | 0.147 |
| weekend_dinner | 0.386 | 52 | 0.425 | 52 | 0.246 | 47 | 0.265 | 47 | 0.133 | 47 | 0.142 | 47 | 0.038 | 52 | 0.063 | 52 | 0.143 | 0.163 |
| weekend_late_night | 0.209 | 52 | 0.239 | 52 | 0.196 | 28 | 0.247 | 25 | 0.072 | 28 | 0.091 | 25 | 0.016 | 52 | 0.027 | 52 | 0.088 | 0.111 |
| ALL DAYPARTS | 0.337 | 416 | 0.388 | 416 | 0.225 | 334 | 0.258 | 323 | 0.111 | 332 | 0.122 | 322 | 0.025 | 416 | 0.047 | 416 | 0.122 | 0.146 |

---

### 5. top_ordered_stores  [exact ID set, per daypart]
*Coverage = |V3 stores ∩ V4 stores| / |V3 stores|*
| Level | IOU | n | Coverage | n |
|---|---:|---:|---:|---:|
| weekday_breakfast | 0.748 | 52 | 0.866 | 52 |
| weekday_lunch | 0.418 | 52 | 0.445 | 52 |
| weekday_dinner | 0.614 | 52 | 0.654 | 52 |
| weekday_late_night | 0.673 | 52 | 0.750 | 52 |
| weekend_breakfast | 0.486 | 52 | 0.561 | 52 |
| weekend_lunch | 0.673 | 52 | 0.745 | 52 |
| weekend_dinner | 0.701 | 52 | 0.754 | 52 |
| weekend_late_night | 0.456 | 52 | 0.524 | 52 |
| ALL DAYPARTS | 0.596 | 416 | 0.662 | 416 |

---

