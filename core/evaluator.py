import numpy as np

def get_observed_entries(matrix: np.ndarray) -> np.ndarray:
    """Return structured array of (user, item, rating) for all non-zero entries."""
    users, items = np.nonzero(matrix)
    dtype = np.dtype([("user", np.uint32), ("item", np.uint32), ("rating", np.float32)])
    out = np.empty(len(users), dtype=dtype)
    out["user"] = users
    out["item"] = items
    out["rating"] = matrix[users, items]
    return out


def evaluate_sim_matrix(mat: np.ndarray, labels, a, b, c):
    """Takes in Similarity matrix and movies to compare, returns the similarity of those movies and the max/min similar movies."""

    print(f"{labels[a]} <-> {labels[b]}: {mat[a,b]:.4}")
    print(f"{labels[a]} <-> {labels[c]}: {mat[a,c]:.4}")
    print(f"{labels[b]} <-> {labels[c]}: {mat[b,c]:.4}")
    min_val = mat.min()
    row_id, col_id = np.unravel_index(mat.argmin(), mat.shape)
    print(f"min sim: {min_val:.4} for {labels[row_id]} <-> {labels[col_id]}")
    copy_mat = mat.copy()
    np.fill_diagonal(copy_mat,-1)
    max_val = copy_mat.max()
    row_id, col_id = np.unravel_index(copy_mat.argmax(), mat.shape)
    print(f"max sim: {max_val:.4} for {labels[row_id]} <-> {labels[col_id]}")


def kfold_split(observed: np.ndarray, k: int, seed: int = 42) -> list:
    """Shuffle and split observed ratings into k equal folds."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(observed))
    rng.shuffle(idx)
    return np.array_split(observed[idx], k)

def predict_single(matrix, user, item, simMatrix, estMethod, simMatrix2 = None, estMethod2 = None, ratio=.5):
    """Safely predict a single rating using your estimator."""
    if simMatrix2 is not None:
        try:
            est1 = estMethod(matrix, user, simMatrix, item)
            est2 = estMethod2(matrix, user, simMatrix2, item)
            return est1*(ratio) + est2*(1-ratio)
        except:
            return 0
    try:
        return estMethod(matrix, user, simMatrix, item)
    except:
        return 0  # fallback for safety
    
def evaluate_user(matrix, user, simMatrix, estMethod, simMatrix2=None, estMethod2=None, test_size: float = 0.2, seed: int = 42):
    """Evaluate a single user's prediction accuracy using an 80/20 train/test split. Returns MAE on the held-out 20% as a confidence measure for that user's recommendations."""
    rated_items = np.nonzero(matrix[user])[0]

    if len(rated_items) < 5:
        print(f"[evaluator] User has only {len(rated_items)} ratings — not enough to evaluate.")
        return None

    rng = np.random.default_rng(seed)
    shuffled = rated_items.copy()
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_size))
    test_items  = shuffled[:n_test]
    # train_items = shuffled[n_test:]  # implicit — everything not zeroed out

    train_matrix = matrix.copy()
    true_ratings = matrix[user, test_items].copy()
    train_matrix[user, test_items] = 0

    errors = []
    for item, true_rating in zip(test_items, true_ratings):
        pred = predict_single(train_matrix, user, item, simMatrix, estMethod, simMatrix2, estMethod2)
        errors.append(abs(pred - true_rating))

    mae = float(np.mean(errors))
    print(f"[evaluator] User MAE: {mae:.4f}  ({n_test} test items, {len(shuffled) - n_test} train items)")
    return mae


def evaluate_fold(matrix, test_entries, simMatrix, estMethod, simMatrix2=None, estMethod2=None, metric="MAE", ratio=0.5):
    """Evaluate MAE on a single fold."""
    errors = []

    # Make a copy so we can hide test ratings
    train_matrix = matrix.copy()

    # Hide test ratings
    for user, item, true_rating in test_entries:
        train_matrix[user, item] = 0

    # Predict each hidden rating
    for user, item, true_rating in test_entries:
        pred = predict_single(train_matrix, user, item, simMatrix, estMethod, simMatrix2, estMethod2, ratio)
        if metric=="MAE":
            errors.append(abs(pred - true_rating))
        elif metric=="Mean Error":
            errors.append(pred - true_rating)
        elif metric=="RMSE":
            errors.append(((pred - true_rating) ** 2))

    if metric == "RMSE":
        result = np.sqrt(np.mean(errors))
    else:
        result = np.mean(errors)
    return result


def kfold_evaluate(
    matrix:      np.ndarray,
    sim_matrix:  np.ndarray,
    est_method,
    k:           int   = 5,
    sample_frac: float = 1.0,
    seed:        int   = 42,
    sim_matrix2 = None,
    est_method2 = None,
    ratio:       float = 0.5,
    eval_method: str   = "MAE"
) -> list:
    """Run k-fold cross-validation and return per-fold MAE scores.
 
    Parameters
    ----------
    matrix      : user-item numpy array (0 = unrated)
    sim_matrix  : precomputed user-user similarity matrix
    est_method  : callable method
    k           : number of folds (default 5)
    sample_frac : fraction of observed entries to use.
                    1.0  — full cross-validation, exact MAE (slow)
                    0.0–1.0 — random subsample, approximate MAE (faster)
                  Example: sample_frac=0.2 uses 20% of entries → ~5x faster,
                  with typically <0.05 MAE difference on a dense matrix.
    seed        : random seed for reproducible splits and sampling
    sim_matrix2 : precomputed user-user similarity matrix 2
    est_method2 : callable method 2
    eval_method : "MAE" or "Mean Error" or "RMSE"
 
    Returns
    -------
    List of k MAE floats. Prints fold-by-fold results and final average.
    """
    if not (0 < sample_frac <= 1.0):
        raise ValueError(f"sample_frac must be between 0 (exclusive) and 1.0 (inclusive), got {sample_frac}")
 
    observed = get_observed_entries(matrix)
 
    # --- Apply sampling if requested ---
    if sample_frac < 1.0:
        rng        = np.random.default_rng(seed)
        n_sample   = max(k, int(len(observed) * sample_frac))  # always keep at least k entries
        sample_idx = rng.choice(len(observed), size=n_sample, replace=False)
        observed   = observed[sample_idx]
 
    folds = kfold_split(observed, k, seed=seed)
 
    mode_str = f"sample_frac={sample_frac:.0%}" if sample_frac < 1.0 else "full"
    print(
        f"[evaluator] {len(observed):,} entries ({mode_str}) | "
        f"{k} folds | ~{len(observed) // k:,} test entries per fold"
    )
 
    fold_results = []
    for i, fold in enumerate(folds):
        result = evaluate_fold(matrix, fold, sim_matrix, est_method, simMatrix2=sim_matrix2, estMethod2=est_method2, metric=eval_method, ratio=ratio)
        print(f"  Fold {i + 1}/{k}  {eval_method}: {result:.4f}  ({len(fold):,} entries)")
        fold_results.append(result)
 
    avg = float(np.mean(fold_results))
    print(f"\n[evaluator] Average {eval_method}: {avg:.4f}  (across {k} folds)")
    return fold_results