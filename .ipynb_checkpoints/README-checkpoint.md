# NBA Player Classification

Unsupervised machine learning project that classifies NBA players into meaningful archetypes using player statistics from the 2023-24 regular season.

![Cluster Radar Charts](figures/09_pca_projection_w_cluster.png)

---

## Overview

Traditional NBA positions (PG, SG, SF, PF, C) fail to capture the full diversity of modern player roles. This project uses KMeans clustering on a rich feature set drawn from multiple NBA API endpoints to discover data-driven player archetypes that better reflect how players actually play.

The final model identifies **12 distinct player types**, ranging from elite offensive bigs to floor spacers, with strong alignment to basketball domain knowledge.

---

## Results

| Cluster | Type |
|---------|------|
| 0 | 3-and-D Wings |
| 1 | Floor Spacers |
| 2 | Rim Runners |
| 3 | Secondary Scoring Guards |
| 4 | Athletic Forwards |
| 5 | Role Players |
| 6 | Backup PGs |
| 7 | Elite Offensive Bigs |
| 8 | Versatile Wings |
| 9 | Backup Bigs |
| 10 | Stretch Bigs |
| 11 | Elite Scoring Guards |

---

## Data

All data is fetched from the [nba_api](https://github.com/swar/nba_api) library using the 2023-24 regular season. Five endpoints are used and merged on `PLAYER_ID`:

| Endpoint | Description |
|----------|-------------|
| `LeagueDashPlayerStats` | Per 36 min base stats (FGA, FG%, AST, etc.) |
| `LeagueDashPlayerBioStats` | Physical attributes and advanced metrics (height, weight, TS%, USG%) |
| `LeagueHustleStatsPlayer` | Hustle stats (screen assists, box outs, loose balls) |
| `LeagueDashPtStats` | Tracking stats (speed, passes, drives) across SpeedDistance, Passing, Drives |
| `LeagueDashPlayerShotLocations` | Shot frequency and efficiency by court zone |

Players with fewer than 10 minutes per game or fewer than 20 games played are excluded, leaving **445 players**.

---

## Pipeline

### 1. Data Collection
- Fetch and merge five nba_api endpoints
- Filter low-minute and low-games players
- Drop redundant, non-offensive, and administrative columns

### 2. EDA
- Distribution histograms for all numeric features
- Correlation matrix to identify redundant features

### 3. Preprocessing

**Null handling:** SimpleImputer with zero fill for missing values.

**Noise Filters:** Three custom filters to suppress statistically unreliable efficiency metrics caused by low sample sizes:

- **FG3 Noise Filter** (`threshold = 0.25`): Players whose three-point attempt rate (`FG3A / FGA`) falls below the threshold have all three-point efficiency columns zeroed out. Prevents rare shooters from inflating FG3%.
- **Mid-Range Noise Filter** (`threshold = 0.0125`): Same logic applied to mid-range shooting zones.
- **Big Man Drive Noise Filter** (sigmoid weighting): Drive columns are down-weighted using a sigmoid function centered at the median player height. This prevents interior finishes by big men from being mistaken for guard-style drives, as NBA drive tracking does not distinguish between these two types of plays.

**Playmaking weighting:** AST-related columns are multiplied by 1.3 to amplify playmaking signal, improving separation of true playmakers (e.g. Jokić, Sabonis) from pure scorers and bigs.

**Standard Scaling:** All features scaled to mean = 0, std = 1 before PCA.

### 4. Dimensionality Reduction (PCA)
- PCA applied to the full scaled feature matrix
- First 6 principal components used for clustering, capturing the majority of variance
- PC1 captures **offensive playmaking load** (Trae Young, Luka, SGA at top; pure rim runners at bottom)
- PC2 captures **interior dominance** (Jokić, Sabonis, Embiid at top; perimeter-only guards at bottom)

### 5. Clustering (KMeans)
- K selected via elbow method and silhouette score
- k=12 chosen as it is a local silhouette maximum (0.2021) with clear basketball interpretability
- Model fit on first 6 PCA components

---

## Evaluation

### Star Player Validation
Known star players are used as a qualitative sanity check. Key results:

- **Cluster 7 (Elite Offensive Bigs):** Jokić, Embiid, Giannis, AD, Sabonis, Adebayo, Sengun
- **Cluster 8 (Versatile Wings):** LeBron, KD, Tatum, Jaylen Brown, KAT, Wembanyama, Kawhi
- **Cluster 11 (Elite Scoring Guards):** Luka, Curry, SGA, Trae, Mitchell, Lillard, Edwards, Kyrie

### Cluster Profile Heatmap
Z-score normalized feature means visualized as a heatmap across all 12 clusters, showing which features distinguish each cluster.

### Cluster Radar Charts
17 representative features across six dimensions — body type, interior scoring, perimeter shooting, offensive load, playmaking, and athleticism — visualized per cluster.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Language |
| Jupyter Notebook | Development environment |
| nba_api | Data source |
| pandas, numpy | Data manipulation |
| matplotlib, seaborn | Visualization |
| scikit-learn | PCA, KMeans, StandardScaler |

---

## Project Structure

```
nba-player-classification/
├── notebook.ipynb          # Main notebook
├── README.md
└── figures/
    ├── 01_correlation_matrix.png
    ├── 02_distribution_histograms.png
    ├── 03_fg3_distribution.png
    ├── 04_mid-range_distribution.png
    ├── 05_sigmoid_drive_weight.png
    ├── 06_pca_screeplot.png
    ├── 07_pca_projection_2d.png
    ├── 08_kmeans_select_k.png
    ├── 09_pca_projection_w_cluster.png
    ├── 10_cluster_profile_heatmap.png
    └── 11_cluster_radar_chart.png
```

---

## Installation

```bash
pip install nba_api pandas numpy matplotlib seaborn scikit-learn
```