# Clan Tournament Match Predictor
### Nordeus Job Fair 2026 – Data Science Challenge

A machine learning solution for predicting clan tournament match winners in a mobile strategy game, plus a bonus **Clan Improvement Advisor** that tells a clan exactly which stats to focus on before facing a specific opponent.

---

## Problem Statement

Given pre-match statistics for each clan's members, predict which clan will win a head-to-head tournament match (`clan_1` or `clan_2`).

Two datasets are provided:
- **`member_stats_*.csv`** – per-player stats for each clan member (activity, training, star ratings, payer status, etc.)
- **`clan_matches_*.csv`** – match pairings and (in training) the outcome

---

## Approach

```
Member stats (player level)
        │
        ▼
Clan-level aggregation (mean / max / min / std / sum)
        │
        ▼
Match-level features (clan_1 − clan_2 differences & ratios)
        │
        ▼
Gradient Boosting Classifier (5-fold CV)
        │
        ▼
Predictions → clan_winner_predictions.csv
```

1. **Aggregate** member stats to clan level using multiple statistics (mean, max, min, std, sum) to capture both average strength and squad depth.
2. **Build match features** as pairwise differences and ratios between the two clans — encoding relative advantage rather than raw values.
3. **Train** a Gradient Boosting Classifier and compare against Random Forest using stratified 5-fold cross-validation.
4. **Generate predictions** for the test set.

---

## Key Findings

| Feature | Why it matters |
|---|---|
| `avg_stars_top_11_players` / `avg_stars_top_3_players` | Overall squad quality — the strongest predictor of match outcome |
| `clan_multiplier` | Directly scales points earned during the tournament |
| `days_active_last_28_days` / `training_count_last_28_days` | Active, training managers consistently outperform inactive ones |
| `avg_training_bonus` | Compounds with activity — high-bonus active players are disproportionately effective |

---

## Repository Structure

```
.
├── final2.ipynb                    # Main notebook (EDA → features → model → predictions)
├── member_stats_training.csv       # Player-level stats for training clans
├── clan_matches_training.csv       # Training match pairings + outcomes
├── member_stats_test.csv           # Player-level stats for test clans
├── clan_matches_test.csv           # Test match pairings (no outcome)
└── clan_winner_predictions.csv     # Generated predictions (output)
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Python 3.8+ recommended.

---

## Usage

Open and run `final2.ipynb` top to bottom. The notebook is structured into self-contained sections:

| Section | Description |
|---|---|
| 1. Load Data | Reads all four CSVs and performs sanity checks |
| 2. EDA | Distributions, class balance, correlation heatmap |
| 3. Feature Engineering | Clan aggregation + match-level diff/ratio features |
| 4. Modeling | GBM vs Random Forest with 5-fold CV |
| 5. Feature Importance | Top 20 drivers visualised |
| 6. Predictions | Outputs `clan_winner_predictions.csv` |
| 7. Bonus: Clan Advisor | Per-matchup improvement recommendations |

---

## Bonus: Clan Improvement Advisor

The `clan_improvement_advisor` function helps a clan understand **where to focus** before a specific matchup.

**How it works:**
- Pulls feature importances from the trained model (what matters globally)
- Computes the gap between the clan's stats and the opponent's (where we're actually behind)
- Ranks improvement areas by `importance × relative_gap` — so only gaps that the model cares about bubble to the top

```python
advice = clan_improvement_advisor(
    clan_id=my_clan_id,
    opponent_clan_id=their_clan_id,
    clan_feats=clan_features_test,
    model=best_model,
    feat_cols=feature_cols
)
```

The output is a ranked table with each improvable stat, the current values for both clans, the gap, and a priority score. A bar chart visualises which areas are red (behind the opponent) vs. blue (already ahead).

---

## Model Performance

Models are evaluated with stratified 5-fold cross-validation. The best-performing model (Gradient Boosting or Random Forest, selected automatically) is then retrained on the full training set before generating test predictions.
