"""Target-decoy false discovery rate (FDR) estimation.

Each item (PSM, peptide or protein) carries a score and a decoy flag. Items
are ranked by score and the empirical FDR at each threshold is
(decoys + pseudocount) / targets. The q-value of an item is the smallest FDR
over all thresholds at or below its rank, so q-values are monotone.
"""
import numpy as np


def qvalues(scores, is_decoy, higher_better=True, pseudocount=1.0):
    """Return target-decoy q-values in the same order as the input.

    Items with tied scores share one threshold, so a target and a decoy with
    the same score are judged together rather than in an arbitrary order.
    """
    scores = np.asarray(scores, dtype=float)
    is_decoy = np.asarray(is_decoy, dtype=bool)
    if scores.shape != is_decoy.shape:
        raise ValueError("scores and is_decoy must have the same length")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN; drop or impute them first")
    n = scores.size
    if n == 0:
        return np.empty(0)

    s = scores if higher_better else -scores
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    decoy_sorted = is_decoy[order]

    targets_cum = np.cumsum(~decoy_sorted)
    decoys_cum = np.cumsum(decoy_sorted)
    fdr = (decoys_cum + pseudocount) / np.maximum(targets_cum, 1)

    # One threshold per distinct score: use the FDR at the last item of each tie group.
    new_group = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    group_id = np.cumsum(new_group) - 1
    group_last = np.r_[np.flatnonzero(new_group)[1:] - 1, n - 1]
    group_q = np.minimum.accumulate(fdr[group_last][::-1])[::-1]

    q = np.empty(n)
    q[order] = group_q[group_id]
    return np.minimum(q, 1.0)
