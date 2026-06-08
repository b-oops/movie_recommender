# 🎬 Movie Recommender

A personalised movie recommendation app built on collaborative filtering. Upload your [Letterboxd](https://letterboxd.com) ratings export and get picks drawn from a curated community of film critics — covering films up to 2023.

Live at: [your Streamlit Cloud URL here]

---

## How it works

The app uses **item-based and user-based k-nearest-neighbor collaborative filtering** to predict how much you'd enjoy films you haven't seen, based on what you and similar users have rated.

### Recommendation pipeline

1. **Upload** — you provide your Letterboxd `ratings.csv` export
2. **Matching** — your rated films are matched to the community dataset by slugifying titles and trying exact + year-suffix matches (e.g. `parasite` → `parasite-2019`)
3. **Matrix construction** — your ratings are appended as a new row to the 938-user × 8048-movie community ratings matrix
4. **Item similarity** — loaded from a precomputed top-N neighbors JSON (`item_topn.json`), built offline using mean-centred cosine similarity across the 938 community users
5. **User similarity** — loaded from a precomputed community user sim matrix (`user_sims.csv`), then your similarities to all 938 community users are computed live using cosine similarity on your rating vector
6. **Scoring** — unrated films are scored using a weighted blend of:
   - `kNearestItemEst_topn` — predicts based on how similar a film is to films you've already rated highly
   - `kNearestUserEst` — predicts based on what users similar to you have rated highly
   - The blend ratio is controlled by the sidebar slider
7. **Results** — top N films displayed with title, year, runtime, genres, and overview

---

## Repo structure

```
movie_recommender/
├── app.py                  # Streamlit app — main entry point
├── requirements.txt
├── core/
│   ├── loader.py           # CSV loading and title slug matching
│   ├── matrix.py           # User-item matrix construction and similarity computation
│   ├── recommender.py      # Similarity measures and estimation functions
│   ├── filters.py          # Movie filtering strategies (metadata + community density)
│   └── evaluator.py        # k-fold cross-validation and per-user MAE evaluation
└── data/
    ├── ratings_filtered.csv        # Filtered community ratings (~938 users, ~8048 movies)
    ├── movie_data_filtered.csv     # TMDB metadata (title, year, genres, runtime, overview)
    ├── item_topn.json              # Precomputed item top-N neighbor similarities
    └── user_sims.csv               # Precomputed 938×938 community user similarity matrix
```

---

## Data files

The `data/` directory is not fully tracked in git due to file size. You need to provide:

| File | Description | How to get it |
|---|---|---|
| `ratings_filtered.csv` | Filtered community ratings | Run your filtering pipeline on the raw Letterboxd community export |
| `movie_data_filtered.csv` | TMDB movie metadata | Included in the repo |
| `item_topn.json` | Precomputed item similarities |
| `user_sims.csv` | Precomputed user similarities |

---

## Precomputing data files

Whenever you update `ratings_filtered.csv`, regenerate the similarity files by running locally:

```bash
python precompute.py
```

This produces `item_topn.json` and `user_sims.csv` in `data/`. It will take several minutes for the item similarity matrix (8048 × 8048 pairs). Commit both files before deploying.

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push the repo to GitHub (ensure `data/` files are committed)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set main file to `app.py`
4. Deploy — you'll get a public URL within a couple of minutes

---

## Settings

| Setting | Description |
|---|---|
| Number of recommendations | How many films to return (3–25) |
| Item ← ratio → User | Blend between item-based (1.0) and user-based (0.0) CF |
| Rewatch mode | If enabled, scores films you've already rated instead of unseen ones |

---

## Community dataset

The community ratings come from a curated set of active Letterboxd critics, filtered to under 50MB:
- Users with at least N ratings -- N chosen as 1000
- Films rated by at least M users -- M chosen as 200
- Films with sufficient TMDB metadata (runtime between 80–240 min, basic vote counts)

Coverage is limited to films released up to **2023**. Newer films will show as unmatched when you upload.

---

## Core modules

### `recommender.py`

- `cosineSim` / `pearsonSim` — similarity measures with significance weighting
- `kNearestItemEst_topn` — item-based kNN prediction using precomputed top-N neighbors; mean-centred to correct for rating bias
- `kNearestUserEst` — user-based kNN prediction
- `weighted_recommend` — scores all unrated films and returns the top N; supports blended item+user estimation and rewatch mode

### `evaluator.py`

- `kfold_evaluate` — k-fold cross-validation over the full matrix; supports MAE, RMSE, and Mean Error metrics
- `evaluate_user` — 80/20 train/test evaluation for a single user; useful as a per-user confidence measure
- `predict_single` — safely wraps a single prediction with fallback

### `filters.py`

- `metadata_filter` — filters movies using TMDB fields (popularity, vote count, genre, language, year, runtime)
- `community_filter` — filters to movies rated by at least X% of community users
- `generate_movie_filter` — combines both strategies; used by the loader

---

## Limitations

- Films released after 2023 will not appear in recommendations or be matchable from your ratings
- The item similarity matrix is computed on the 938 community users only; your ratings do not update it
- Match quality depends on Letterboxd title slugs — some films with special characters or disambiguation suffixes may not match