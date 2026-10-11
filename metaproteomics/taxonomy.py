"""Taxonomic assignment of peptides by lowest common ancestor (LCA)."""
import pandas as pd


def read_taxonomy(path, sep="\t"):
    """Read a child-to-parent table with columns ``taxid`` and ``parent``.

    Returns a dict mapping taxid to parent taxid. Rows with an empty parent
    become roots.
    """
    df = pd.read_csv(path, sep=sep, dtype="Int64")
    return {int(t): (int(p) if pd.notna(p) else None)
            for t, p in zip(df["taxid"], df["parent"])}


def lineage(taxid, parents, max_depth=100):
    """Return the chain from ``taxid`` up to its root, stopping on cycles."""
    chain, seen, t = [], set(), taxid
    while t is not None and t not in seen and len(chain) < max_depth:
        chain.append(t)
        seen.add(t)
        t = parents.get(t)
    return chain


def lca(taxids, parents):
    """Return the lowest common ancestor of ``taxids``.

    Unknown taxids (None) are ignored. A single taxid is returned as is. If
    the taxa share no known ancestor, the result is None.
    """
    taxids = {t for t in taxids if t is not None}
    if not taxids:
        return None
    if len(taxids) == 1:
        return next(iter(taxids))
    chains = [list(reversed(lineage(t, parents))) for t in taxids]
    common = None
    for level in zip(*chains):
        if all(x == level[0] for x in level):
            common = level[0]
        else:
            break
    return common
