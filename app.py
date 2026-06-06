import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from core.loader import load_metadata
from core.matrix import build_user_item_df, mean_centre
from core.recommender import standItemEst, weighted_recommend

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
COMMUNITY_PATH  = os.path.join(DATA_DIR, "ratings_filtered.csv")
ITEM_SIM_PATH     = os.path.join(DATA_DIR, "item_sims_preFilter_under50MB.csv")    # precomputed item sims
USER_SIM_PATH     = os.path.join(DATA_DIR, "user_sims_preFilter_under50MB.csv.csv")    # precomputed item sims
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
def load_item_sim():
    """Load precomputed item similarity matrix. Returns (np.ndarray, list of movie_ids)."""
    df = pd.read_csv(ITEM_SIM_PATH, index_col=0)
    return df.to_numpy(dtype=np.float64), list(df.columns)

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
    top_n = st.slider("Number of recommendations", 3, 25, 10)
    st.markdown("---")
    st.markdown(
        "Export your ratings from **Letterboxd → Settings → Import & Export → Export Your Data**. "
        "Upload the `ratings.csv` file."
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

# build matrix
with st.spinner("Building user-item matrix…"):
    combined = pd.concat([community_df, user_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    df_matrix = build_user_item_df(combined)

user_id = "me"
if user_id not in df_matrix.index:
    st.error("Something went wrong adding your ratings to the matrix.")
    st.stop()

matrix_num = np.nan_to_num(np.array(df_matrix), nan=0.0)
centered_num = np.nan_to_num(np.array(mean_centre(np.array(df_matrix))), nan=0.0)

# build similarity matrix (cached by matrix content)
with st.spinner("Loading item similarities…"):
    item_sim_matrix, sim_movie_ids = load_item_sim()

# run recommender
my_user_index = df_matrix.index.get_loc(user_id)
unrated_mask  = matrix_num[my_user_index] == 0

with st.spinner("Generating recommendations…"):
    raw_recs = weighted_recommend(
        matrix    = matrix_num,
        user      = my_user_index,
        simMatrix = item_sim_matrix,
        N         = top_n,
        estMethod = standItemEst,
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
            st.markdown(f"**{i}. {rec['title']}**{year_str}")
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
