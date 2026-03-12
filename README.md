# Recommendation Comparison System

## Overview
This script evaluates recommendation systems A, B, C against food order history to determine their effectiveness in making relevant food suggestions.

## Quick Start
To use the script with default settings, simply run:

```bash
python recommendation_comparison.py
```

## Advanced Mode
For advanced users, you can specify parameters as follows:

```bash
python recommendation_comparison.py --carousels-csv <path_to_carousels.csv> --orders-csv <path_to_orders.csv> --day-part <day_part> --orders-day-part <orders_day_part>
```

## Detailed Command-Line Arguments
- `--carousels-csv`: Path to the carousels CSV file.
- `--orders-csv`: Path to the orders CSV file.
- `--day-part`: The part of the day (e.g., breakfast, lunch, dinner).
- `--orders-day-part`: Specific orders' day part for filtering results.

## How It Works
The script employs a scoring algorithm that ranks the recommendations based on user interactions and historical data. It compares the output from each recommendation system and selects the one with the highest cumulative score.

## Output
The script produces a report detailing the effectiveness of each recommendation system, including metrics and visualizations of their performance against the food order history.