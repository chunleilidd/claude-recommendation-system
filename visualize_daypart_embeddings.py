#!/usr/bin/env python3
"""
visualize_daypart_embeddings.py

Loads 2026-03-13 11_22pm.csv, reduces NORM_CAROUSEL_EMB to 2-D with UMAP,
and renders an interactive Plotly scatter plot coloured by DAY_PART.

Dependencies:
    pip install umap-learn plotly pandas numpy
"""

import json
import ast
import numpy as np
import pandas as pd
import plotly.express as px
import umap

CSV_PATH = "2026-03-13 11_22pm.csv"
OUTPUT_HTML = "daypart_embeddings.html"

# UMAP hyper-parameters
UMAP_N_NEIGHBORS = 10    # small dataset (63 rows) → lower value
UMAP_MIN_DIST    = 0.05
UMAP_METRIC      = "cosine"
RANDOM_STATE     = 42


def parse_embedding(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        # Try JSON first, then ast.literal_eval as fallback
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
    return None


def main():
    # ── 1. Load CSV ───────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows.")

    # ── 2. Parse embeddings ───────────────────────────────────────────────────
    df["embedding"] = df["NORM_CAROUSEL_EMB"].apply(parse_embedding)
    bad = df["embedding"].isna().sum()
    if bad:
        print(f"Warning: {bad} rows had unparseable embeddings — dropping.")
    df = df.dropna(subset=["embedding"]).reset_index(drop=True)

    X = np.array(df["embedding"].tolist(), dtype=np.float32)
    print(f"Embedding matrix shape: {X.shape}")

    # ── 3. UMAP ───────────────────────────────────────────────────────────────
    print(f"Running UMAP (n_neighbors={UMAP_N_NEIGHBORS}, "
          f"min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC})...")
    reducer = umap.UMAP(
        n_components = 2,
        n_neighbors  = UMAP_N_NEIGHBORS,
        min_dist     = UMAP_MIN_DIST,
        metric       = UMAP_METRIC,
        random_state = RANDOM_STATE,
        verbose      = False,
    )
    coords = reducer.fit_transform(X)
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]

    # ── 4. Order day-parts chronologically ───────────────────────────────────
    daypart_order = [
        "weekday_breakfast", "weekend_breakfast",
        "weekday_lunch",     "weekend_lunch",
        "weekday_dinner",    "weekend_dinner",
        "weekday_late_night","weekend_late_night",
    ]
    # Keep only those present in the data, preserving order
    present = df["DAY_PART"].unique().tolist()
    daypart_order = [d for d in daypart_order if d in present] + \
                    [d for d in present if d not in daypart_order]

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    fig = px.scatter(
        df,
        x               = "umap_x",
        y               = "umap_y",
        color           = "DAY_PART",
        category_orders = {"DAY_PART": daypart_order},
        hover_name      = "TITLE",
        hover_data      = {
            "DAY_PART":      True,
            "CAROUSEL_RANK": True,
            "THEME":         True,
            "CONSUMER_ID":   True,
            "umap_x":        False,
            "umap_y":        False,
        },
        symbol          = "DAY_PART",
        opacity         = 0.82,
        title           = "Carousel Embedding Landscape — coloured by Day Part (UMAP of NORM_CAROUSEL_EMB)",
        labels          = {"umap_x": "UMAP-1", "umap_y": "UMAP-2",
                           "DAY_PART": "Day part"},
        template        = "plotly_white",
        width           = 1100,
        height          = 800,
        color_discrete_sequence = px.colors.qualitative.Bold,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))
    fig.update_layout(
        legend=dict(title="Day part", itemsizing="constant"),
        font=dict(size=13),
    )

    fig.write_html(OUTPUT_HTML)
    print(f"Saved → {OUTPUT_HTML}")

    try:
        fig.write_image("daypart_embeddings.png", scale=2)
        print("Saved → daypart_embeddings.png")
    except Exception as e:
        print(f"PNG skipped ({e}). Install kaleido: pip install kaleido")

    print("\nDone. Open daypart_embeddings.html in your browser.")


if __name__ == "__main__":
    main()
