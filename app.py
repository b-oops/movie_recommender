import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from core.loader import load_metadata
from core.matrix import build_user_item_df, mean_centre
from core.recommender import kNearestItemEst_topn, kNearestUserEst, weighted_recommend

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
COMMUNITY_PATH  = os.path.join(DATA_DIR, "ratings_filtered.csv")
ITEM_SIM_PATH     = os.path.join(DATA_DIR, "item_topn.json")    # precomputed item sims
USER_SIM_PATH     = os.path.join(DATA_DIR, "user_sims.csv")    # precomputed item sims
METADATA_PATH   = os.path.join(DATA_DIR, "movie_data_filtered.csv")

# ── cached data loaders ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_community():
    df = pd.read_csv(COMMUNITY_PATH)
    df = df.rename(columns={"rating_val": "rating"})
    df["rating"] = df["rating"].astype(float)
    return df

@st.cache_data(show_spinner=False)
def load_meta():
    return load_metadata(METADATA_PATH)

@st.cache_data(show_spinner=False)
def load_user_sim():
    """Load precomputed community user similarity matrix."""
    df = pd.read_csv(USER_SIM_PATH, index_col=None, header=None)
    return df.to_numpy(dtype=np.float64)

@st.cache_data(show_spinner=False)
def load_item_topn() -> dict:
    """Load item_topn.json. Returns dict of {item_index: [[neighbor_index, sim], ...]}."""
    import json
    with open(ITEM_SIM_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

# ── helpers ────────────────────────────────────────────────────────────────
def parse_uploaded_ratings(uploaded_file, known_ids: set) -> pd.DataFrame:
    """Parse user-uploaded Letterboxd CSV into [user_id, movie_id, rating]."""
    import re
    df = pd.read_csv(uploaded_file)

    def slugify(title: str) -> str:
        title = str(title).lower().strip()
        title = re.sub(r"[^a-z0-9\s-]", "", title)
        title = re.sub(r"\s+", "-", title)
        return title.strip("-")

    def match_slug(slug, year):
        if slug in known_ids:
            return slug
        slug_year = f"{slug}-{int(year)}" if pd.notna(year) else slug
        if slug_year in known_ids:
            return slug_year
        return None

    df["slug"]     = df["Name"].apply(slugify)
    df["movie_id"] = df.apply(lambda r: match_slug(r["slug"], r.get("Year")), axis=1)

    matched   = df["movie_id"].notna()
    unmatched = (~matched).sum()

    df           = df[matched].copy()
    df["user_id"] = "me"
    df["rating"]  = df["Rating"] * 2   # 0.5–5 → 1–10

    return df[["user_id", "movie_id", "rating"]], unmatched


# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🎬 Movie Recommender")
st.caption("Upload your Letterboxd ratings export and get personalised picks from top letterboxd critics.")

# sidebar
with st.sidebar:
    st.header("Settings")
    top_n   = st.slider("Number of recommendations", 3, 25, 10)
    ratio   = st.slider("Based on OTHER users' favorites ← ratio → based on YOUR favorites", 0.0, 1.0, 0.5, step=0.05)
    rewatch = st.toggle("Rewatch mode", value=False)
    run     = st.button("Get recommendations", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown(
        "Export your ratings from **Letterboxd → Settings → Import & Export → Export Your Data**. "
        "Upload the `ratings.csv` file. [Open Letterboxd ↗](https://letterboxd.com) "
        "Letterboxd is free to use and the best thing in the world. Don't be scared."
    )

# file upload
uploaded = st.file_uploader("Upload your Letterboxd ratings CSV", type="csv")

if not uploaded:
    st.info("👆 Upload your ratings CSV to get started.")
    st.stop()

# load community data
with st.spinner("Loading community ratings…"):
    community_df = load_community()
    metadata_df  = load_meta()

known_ids = set(community_df["movie_id"].unique())

# parse user file
user_df, unmatched = parse_uploaded_ratings(uploaded, known_ids)

if len(user_df) == 0:
    st.error("No movies from your ratings matched our database. Make sure you're uploading the Letterboxd `ratings.csv` export.")
    st.stop()

matched_count = len(user_df)
avg_rating    = (user_df["rating"] / 2).mean()   # back to 0.5–5 scale for display

col1, col2, col3 = st.columns(3)
col1.metric("Ratings matched", matched_count)
col2.metric("Avg rating", f"{avg_rating:.2f} ★")
col3.metric("Unmatched", unmatched)

if unmatched > 0:
    st.caption(f"{unmatched} ratings couldn't be matched — likely newer films not yet in the community dataset. (we only have data up to 2023)")

st.divider()

if not run:
    st.info("👈 Adjust settings and hit **Get recommendations** to generate your picks.")
    st.stop()

# build matrix
with st.spinner("Building user-item matrix…"):
    combined = pd.concat([community_df, user_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    df_matrix = build_user_item_df(combined)

user_id = "me"
if user_id not in df_matrix.index:
    st.error("Something went wrong adding your ratings to the matrix.")
    st.stop()

# move user to bottom of df
my_user_index = df_matrix.index.get_loc(user_id)
row = df_matrix.iloc[[my_user_index]]
df_matrix = pd.concat(
    [df_matrix.drop(df_matrix.index[my_user_index]), row],
    axis=0
)

matrix_num = np.nan_to_num(np.array(df_matrix), nan=0.0)
centered_num = np.nan_to_num(np.array(mean_centre(np.array(df_matrix))), nan=0.0)

# load precomputed item top-N neighbours
with st.spinner("Loading item similarities…"):
    item_topn = load_item_topn()

# load precomputed community user sim matrix and extend with new user's similarities
my_user_index = df_matrix.index.get_loc(user_id)
with st.spinner("Computing your user similarities…"):
    community_sim = load_user_sim()                      # (938 × 938)
    from core.recommender import cosineSim

    # community rows only (exclude the new user at my_user_index)
    n = community_sim.shape[0]
    community_matrix = np.delete(matrix_num, my_user_index, axis=0)
    #community_matrix = matrix_num[:n]        # all rows except last
    new_user_vec     = matrix_num[my_user_index]         # new user row

    # compute similarity between new user and each community user
    new_user_sims = np.array([
        cosineSim(new_user_vec, community_matrix[u])
        for u in range(n)
    ])

    # extend community sim matrix: add new row and column for the new user
    user_sim_matrix = np.full((n + 1, n + 1), 0.5, dtype=np.float64)
    user_sim_matrix[:n, :n] = community_sim              # community sims unchanged
    user_sim_matrix[n, :n]  = new_user_sims              # new user → community
    user_sim_matrix[:n, n]  = new_user_sims              # community → new user (symmetric)
    np.fill_diagonal(user_sim_matrix, 1.0)

# run recommender
my_user_index = df_matrix.index.get_loc(user_id)
unrated_mask  = matrix_num[my_user_index] == 0

with st.spinner("Generating recommendations…"):
    raw_recs = weighted_recommend(
        matrix     = matrix_num,
        user       = my_user_index,
        simMatrix  = item_topn,
        N          = top_n,
        ratio      = ratio,
        rewatch    = rewatch,
        estMethod  = kNearestItemEst_topn,
        simMatrix2 = user_sim_matrix,
        estMethod2 = kNearestUserEst,
    )

if isinstance(raw_recs, str):
    st.success("You've rated everything in our database — impressive!")
    st.stop()

# map indices → movie_ids
movie_labels = df_matrix.columns
rec_movie_ids   = [movie_labels[idx] for idx, _ in raw_recs]
rec_scores      = [score for _, score in raw_recs]

# enrich with metadata
meta_lookup = metadata_df.set_index("movie_id")

results = []
for movie_id, score in zip(rec_movie_ids, rec_scores):
    row = meta_lookup.loc[movie_id] if movie_id in meta_lookup.index else None
    results.append({
        "movie_id":    movie_id,
        "title":       row["movie_title"] if row is not None else movie_id,
        "year":        int(row["year_released"]) if row is not None and pd.notna(row["year_released"]) else None,
        "genres":      row["genres"] if row is not None else "",
        "overview":    row["overview"] if row is not None else "",
        "runtime":     int(row["runtime"]) if row is not None and pd.notna(row["runtime"]) else None,
        "tmdb_link":   row["tmdb_link"] if row is not None and pd.notna(row["tmdb_link"]) else None,
        "score":       score,
    })

# ── render results ─────────────────────────────────────────────────────────
st.subheader(f"Your top {top_n} picks")

max_score = max(r["score"] for r in results)
min_score = min(r["score"] for r in results)
score_range = max_score - min_score if max_score != min_score else 1

for i, rec in enumerate(results, 1):
    pct = (((rec["score"] - min_score) / score_range) / 5) + .8  ## shifted to be between 80% and 100% matches

    with st.container(border=True):
        left, right = st.columns([6, 1])
        with left:
            year_str = f" ({rec['year']})" if rec["year"] else ""
            runtime_str = f" · {rec['runtime']} min" if rec["runtime"] else ""
            title_md = f"[{rec['title']}]({rec['tmdb_link']})" if rec["tmdb_link"] else rec["title"]
            st.markdown(f"**{i}. {title_md}**{year_str}{runtime_str}")
            if rec["genres"]:
                try:
                    import ast
                    genres = ast.literal_eval(rec["genres"])
                    st.caption(" · ".join(genres[:4]))
                except Exception:
                    st.caption(str(rec["genres"]))
            if rec["overview"]:
                st.caption(rec["overview"][:200] + ("…" if len(rec["overview"]) > 200 else ""))
        with right:
            st.metric("Match", f"{pct:.0%}")
