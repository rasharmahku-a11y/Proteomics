"""Protein-level FDR (picked target-decoy) and protein grouping."""
import pandas as pd

from .fdr import qvalues


def picked_protein_fdr(scores, decoy_prefix, pseudocount=1.0):
    """Score proteins by picked target-decoy competition.

    ``scores`` maps accession to score. Each target is paired with its decoy
    (``prefix + accession``). Only the higher-scoring member of each pair
    survives; ties are dropped. Unpaired proteins survive as they are. The
    survivors are then assigned q-values.

    Picking avoids the bias of counting every decoy hit, which matters when
    the database is large and many decoy proteins have near-identical targets.
    """
    keys = {a[len(decoy_prefix):] if a.startswith(decoy_prefix) else a
            for a in scores}
    rows = []
    for t in keys:
        ts = scores.get(t)
        ds = scores.get(decoy_prefix + t)
        if ts is not None and ds is not None:
            if ts == ds:
                continue
            rows.append((t, ts, False) if ts > ds else (decoy_prefix + t, ds, True))
        elif ts is not None:
            rows.append((t, ts, False))
        else:
            rows.append((decoy_prefix + t, ds, True))

    df = pd.DataFrame(rows, columns=["accession", "score", "is_decoy"])
    if df.empty:
        df["q_protein"] = pd.Series(dtype=float)
        return df
    df["q_protein"] = qvalues(df["score"], df["is_decoy"], pseudocount=pseudocount)
    return df


def greedy_groups(peptide_proteins, protein_scores):
    """Choose representative proteins by greedy set cover.

    Proteins are visited from best to worst score. A protein is kept only if
    it explains at least one peptide that no kept protein already explains.
    Each peptide is assigned to the first kept protein that covers it.

    Returns a list of dicts, one per representative, with the number of
    peptides assigned to it (``unique``) and the number it shares with other
    proteins (``shared``).
    """
    membership = {}
    for pep, accs in peptide_proteins.items():
        for a in accs:
            membership.setdefault(a, set()).add(pep)

    uncovered = set(peptide_proteins)
    assigned = {}
    reps = []
    for acc in sorted(protein_scores, key=lambda a: -protein_scores[a]):
        new = membership.get(acc, set()) & uncovered
        if not new:
            continue
        for pep in new:
            assigned[pep] = acc
        uncovered -= new
        reps.append(acc)

    results = []
    for acc in reps:
        unique = sum(1 for p, a in assigned.items() if a == acc)
        total = len(membership[acc])
        results.append({"accession": acc, "score": protein_scores[acc],
                        "unique_peptides": unique,
                        "shared_peptides": total - unique})
    return results
