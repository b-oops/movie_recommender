
import pandas as pd
import numpy as np


def build_user_item_df(ratings: pd.DataFrame):
    """Pivot the long-form ratings table into a wide user-item matrix.

    Parameters
    ----------
    ratings : DataFrame with columns [user_id, movie_id, rating]

    Returns
    -------
    DataFrame  shape (n_users, n_movies)
               index = user_id, columns = movie_id, values = rating or NaN
    """
    df = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",   # handles any accidental duplicates gracefully
    )
    df.columns.name = None   # cosmetic: remove the axis label
    df.index.name = "user_id"

    print(
        f"[preprocessor] Matrix shape: {df.shape[0]} users × "
        f"{df.shape[1]} movies. "
        f"Density: {df.notna().sum().sum() / df.size:.2%}"
    )

    return df


def mean_centre(matrix: np.ndarray) -> np.ndarray:
    """Subtract each user's mean rating from their row.

    This corrects for rating bias: a user who gives everything 8/10 and one
    who gives everything 4/10 may have identical taste, but raw cosine
    similarity would say they're very different.

    NaN cells stay NaN — we only centre on observed ratings.
    """
    # Compute row means ignoring NaNs
    row_means = np.nanmean(matrix, axis=1, keepdims=True)

    # Subtract with broadcasting
    centred = matrix - row_means

    return centred


def get_user_vector(df: pd.DataFrame, user_id: str) -> pd.Series:
    """Return a single user's row from the matrix.

    Raises KeyError with a helpful message if user_id is not present.
    """
    if user_id not in df.index:
        raise KeyError(
            f"User '{user_id}' not found in the matrix. "
            f"Available users (sample): {list(df.index[:5])}"
        )
    return df.loc[user_id]


def get_rated_movies(df: pd.DataFrame, user_id: str) -> set:
    """Return the set of movie_ids the given user has already rated."""
    row = get_user_vector(df, user_id)
    return set(row.index[row.notna()])


def get_unrated_movies(df: pd.DataFrame, user_id: str) -> list:
    """Return movie_ids the user has NOT yet rated, in column order."""
    row = get_user_vector(df, user_id)
    return list(row.index[row.isna()])


def get_sim_matrix(matrix, simMeas, dim=1):
    """Compute a symmetric similarity matrix for users or items."""

    if dim == 0:
        mat = matrix
    else:
        mat = matrix.T

    n = mat.shape[0]
    simMatrix = np.zeros((n, n), dtype=float)

    # Compute only upper triangle, then mirror
    for i in range(n):
        simMatrix[i, i] = 1.0  # self-similarity
        for j in range(i + 1, n):
            # Mask out missing values (0 = missing in baseline)
            sim = simMeas(mat[i], mat[j])

            simMatrix[i, j] = sim
            simMatrix[j, i] = sim  # symmetry

    return simMatrix
