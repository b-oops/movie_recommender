"""
filters.py

Reduces the set of movies considered by the recommender before the
user-item matrix is built. Filtering here keeps the matrix small and
focused without needing a sparse representation.

Two independent strategies are provided:

  1. metadata_filter()   — uses movie_data.csv (TMDB popularity, vote
                           count, vote average, genre, language, year).
                           Good when you want to tune *what kinds* of
                           films appear in recommendations.

  2. community_filter()  — uses only the community ratings file itself.
                           Keeps movies that at least N% of users have
                           rated. Good as a quick baseline with no
                           external data needed.

Both return a plain Python set of movie_ids that can be passed to
loader.load_all_ratings() to restrict the matrix columns.

Usage
-----
    from core.filters import metadata_filter, community_filter

    # Option 1 — external metadata
    movie_ids = metadata_filter(
        metadata_path="data/movie_data.csv",
        min_popularity=5.0,
        min_vote_count=50,
    )

    # Option 2 — community coverage only
    movie_ids = community_filter(
        community_path="data/ratings_export_reduced.xlsx",
        min_user_pct=0.05,   # at least 5% of users must have rated it
    )
"""

import ast
import pandas as pd

MOVIE_METADATA_PATH = ""

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _load_metadata(path: str) -> pd.DataFrame:
    """Read movie_data.csv, skipping malformed rows."""
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["movie_id"])
    df["movie_id"] = df["movie_id"].astype(str).str.strip()
    df["year_released"] = df["year_released"].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Strategy 1 — external metadata filter
# ---------------------------------------------------------------------------

def metadata_filter(
    metadata_path:    str,
    min_popularity:   float        = 0.0,
    min_vote_count:   int          = 0,
    min_vote_average: float        = 0.0,
    genres:           list[str]    = None,
    languages:        list[str]    = None,
    year_range:       tuple[int, int] = None,
    runtime_range:       tuple[float, float] = None,
) -> set:
    """Filter movies using TMDB metadata fields.

    Parameters
    ----------
    metadata_path    : path to movie_data.csv
    min_popularity   : TMDB popularity score floor (p50=0.87, p90=4.4,
                       p95=7.7 — so 5.0 keeps the top ~12% by popularity)
    min_vote_count   : minimum number of TMDB votes (filters out
                       near-unrated films; p90 is ~36, p95 is ~113)
    min_vote_average : minimum average TMDB rating (0–10); 0.0 = no filter
    genres           : whitelist of genres e.g. ['Drama', 'Thriller'].
                       None = all genres allowed.
    languages        : whitelist of ISO language codes e.g. ['en', 'fr'].
                       None = all languages allowed.
    year_range       : (min_year, max_year) inclusive. None = no filter.

    Returns
    -------
    set of movie_id strings that pass all filters
    """
    df = _load_metadata(metadata_path)

    # -- numeric filters --
    mask = pd.Series(True, index=df.index)

    if min_popularity > 0:
        mask &= df["popularity"].fillna(0) >= min_popularity

    if min_vote_count > 0:
        mask &= df["vote_count"].fillna(0) >= min_vote_count

    if min_vote_average > 0:
        mask &= df["vote_average"].fillna(0) >= min_vote_average

    # -- year filter --
    if year_range is not None:
        lo, hi = year_range
        mask &= df["year_released"].fillna(0).between(lo, hi)

    # -- runtime filter --
    if runtime_range is not None:
        lo, hi = runtime_range
        mask &= df["runtime"].fillna(0).between(lo, hi)

    # -- language filter --
    if languages is not None:
        lang_set = {l.lower() for l in languages}
        mask &= df["original_language"].str.lower().isin(lang_set)

    # -- genre filter --
    if genres is not None:
        genre_set = {g.lower() for g in genres}

        def _has_genre(val) -> bool:
            if pd.isna(val):
                return False
            try:
                parsed = ast.literal_eval(val)
                return any(g.lower() in genre_set for g in parsed)
            except Exception:
                return False

        mask &= df["genres"].apply(_has_genre)

    filtered = df[mask]
    result   = set(filtered["movie_id"].unique())

    print(
        f"[filters] metadata_filter: {len(result):,} movies pass selected filters."
        f" Dropped {df['movie_id'].nunique() - len(result):,} invalid movies."
        f" Filters — "
        + (f", popularity>={min_popularity}" if (min_popularity>0) else "")
        + (f", votes>={min_vote_count}" if (min_vote_count>0) else "")
        + (f", genres={genres}" if genres else "")
        + (f", languages={languages}" if languages else "")
        + (f", years={year_range}" if year_range else "")
        + (f", runtime={runtime_range}" if runtime_range else "")
    )

    return result


# ---------------------------------------------------------------------------
# Strategy 2 — community coverage filter
# ---------------------------------------------------------------------------

def community_filter(
    df: pd.DataFrame,
    min_user_pct:   float = 0.05,
) -> set:
    """Filter to movies rated by at least `min_user_pct` of community users.

    Parameters
    ----------
    community_path : path to ratings_export_reduced.xlsx
    min_user_pct   : fraction of total users who must have rated the movie
                     e.g. 0.05 = at least 5% of users (slider: 0.01–0.20)

    Returns
    -------
    set of movie_id strings that meet the coverage threshold
    """

    n_users    = df["user_id"].nunique()
    threshold  = int(n_users * min_user_pct)

    counts     = df.groupby("movie_id")["user_id"].nunique()
    qualifying = counts[counts >= threshold]
    result     = set(qualifying.index)

    print(
        f"[filters] community_filter: {len(result):,} movies rated by "
        f">= {min_user_pct:.2%} of {n_users} users (threshold: {threshold} users). "
        f"Dropped {df['movie_id'].nunique() - len(result):,} niche movies."
    )

    return result

## the outward facing function we will call whenever we want to filter a df of movie ratings
def generate_movie_filter(
    df: str,
    method: str,
    metadata: str = ""
        
) -> set:
    """Uses the filter method given by the user to get the set 

    Parameters
    ----------
    community_path : dataframe of ratings
    method :    "standard" -> use only the community rating file to filter out movies
                "strict" -> filter on both

    Returns
    -------
    set of movie_id strings that meet the coverage threshold
    """

    if method == "pre_filter_standard":
        community_density_filter = community_filter(df, min_user_pct=0.08)
        meta_data_filter = metadata_filter(metadata, runtime_range=[80,240])
        combined = community_density_filter & meta_data_filter
        #combined = community_density_filter - meta_data_filter
        return combined
    elif method == "user_standard":
        return metadata_filter(df, metadata, min_popularity=5.0, min_vote_count=50, runtime_range=[80,240])