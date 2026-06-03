import re
import pandas as pd
from core.filters import generate_movie_filter

MY_USER_ID = "b_oops"
MOVIE_METADATA_PATH = ""

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _slugify(title: str) -> str:
    """Convert a movie title to a Letterboxd-style slug.

    'The Dark Knight' -> 'the-dark-knight'
    'Pokémon: The First Movie' -> 'pokmon-the-first-movie'  (non-ASCII stripped)
    """
    title = title.lower().strip()
    title = re.sub(r"[^a-z0-9\s-]", "", title)   # strip special chars
    title = re.sub(r"\s+", "-", title)             # spaces to hyphens
    return title.strip("-")


def _match_slug(slug: str, year: int, known_ids: set) -> str | None:
    """Try to find a movie_id in the community dataset.

    method:
      1. Exact slug match             e.g. 'parasite'
      2. Slug + year suffix           e.g. 'parasite-2019'
    Returns None if no match found.
    """
    if slug in known_ids:
        return slug
    slug_year = f"{slug}-{year}"
    if slug_year in known_ids:
        return slug_year
    return None

def load_metadata(path: str) -> pd.DataFrame:
    """Read movie_data.csv, skipping malformed rows."""
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["movie_id"])
    df["movie_id"] = df["movie_id"].astype(str).str.strip()
    df["year_released"] = df["year_released"].astype("Int64")
    return df


def load_user_ratings(filepath: str, known_ids: set) -> pd.DataFrame:
    """Load the user's Letterboxd CSV export. Returns a DataFrame with columns: user_id, movie_id, rating
    Only rows where a movie_id match is found.
    Unmatched rows are printed as a warning so the caller can track coverage.
    """
    df = pd.read_csv(filepath)#, lineterminator='\n')

    df["slug"] = df["Name"].apply(_slugify)
    df["movie_id"] = df.apply(
        lambda r: _match_slug(r["slug"], r["Year"], known_ids), axis=1
    )

    matched = df["movie_id"].notna()
    unmatched_count = (~matched).sum()
    if unmatched_count:
        print(
            f"[loader] {unmatched_count}/{len(df)} of your ratings could not be "
            f"matched to the community dataset (likely newer films)."
        )

    df = df[matched].copy()
    df["user_id"] = "b_oops"
    df["rating"] = df["Rating"] * 2          # 0.5–5 → 1–10

    return df[["user_id", "movie_id", "rating"]]

def load_community_ratings(filepath: str, filter_method: str) -> pd.DataFrame:
    """Load the community ratings spreadsheet.

    Returns a DataFrame with columns: user_id, movie_id, rating
    Ratings are already on a 1–10 scale.
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={"rating_val": "rating"})
    df["rating"] = df["rating"].astype(float)

    if filter_method != "none":
            before = len(df)
            movie_id_filter = generate_movie_filter(df, filter_method, metadata=MOVIE_METADATA_PATH)
            df = df[df["movie_id"].isin(movie_id_filter)]
            print(
                f"[loader] after all filters applied: kept {len(df):,} of "
                f"{before:,} community ratings "
                f"({df['movie_id'].nunique():,} movies)."
            )

    return df[["user_id", "movie_id", "rating"]]


def load_all_ratings(user_ratings_path: str, community_path: str, method: str = "pre_filter_standard") -> pd.DataFrame:
    """Load and combine both sources into one normalised DataFrame.

    Returns a single DataFrame with columns: user_id, movie_id, rating
    The user's ratings (user_id == 'me') are appended as row 0..N of the
    community matrix so they participate in similarity calculations.
    """
    community = load_community_ratings(community_path, filter_method=method)
    known_ids = set(community["movie_id"].unique())

    user_ratings = load_user_ratings(user_ratings_path, known_ids)

    combined = pd.concat([community, user_ratings], ignore_index=True)

    ## Drop any duplicates (same user rating the same movie twice)
    combined = combined.drop_duplicates(subset=["user_id", "movie_id"], keep="last")

    print(
        f"[loader] Loaded {len(user_ratings)} of your ratings + "
        f"{len(community)} community ratings across "
        f"{combined['movie_id'].nunique()} unique movies and "
        f"{combined['user_id'].nunique()} users."
    )

    return combined

def summarize_movies(metadata: pd.DataFrame, movies: list) -> pd.DataFrame:
    """
    Takes a list of movies and returns relevant columns.
    """

    info = metadata[metadata["movie_id"].isin(movies)]
    info = info[["movie_title", "genres", "original_language", "runtime", "year_released", "overview"]].reset_index(drop = True)

    return info
