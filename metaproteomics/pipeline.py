"""End-to-end metaproteomics FDR pipeline.

Steps:
  1. Check the FASTA decoy set (sizes must match, decoys must be present).
  2. PSM-level q-values; keep PSMs at the requested FDR.
  3. Peptide-level q-values from the best PSM of each peptide.
  4. Protein-level picked-protein FDR, using peptides that pass step 3.
  5. Protein groups by greedy set cover; peptides assigned LCA taxa.
  6. Write tables and a summary with the empirical FDR at each level.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .fdr import qvalues
from .inputs import DEFAULT_DECOY_PREFIX, load_psms, read_fasta
from .proteins import greedy_groups, picked_protein_fdr
from .taxonomy import lca, read_taxonomy


def check_database(entries, decoy_prefix):
    """Count target and decoy entries and return any warnings."""
    decoys = sum(e.accession.startswith(decoy_prefix) for e in entries)
    targets = len(entries) - decoys
    warnings = []
    if decoys == 0:
        warnings.append("no decoy entries found; FDR cannot be estimated")
    elif decoys != targets:
        warnings.append(
            f"decoy count ({decoys}) differs from target count ({targets}); "
            "target-decoy FDR assumes equal-sized, equally competitive decoys")
    return {"targets": targets, "decoys": decoys, "warnings": warnings}


def aggregate_peptides(psms):
    """One row per peptide, taking its best PSM score and decoy flag."""
    ranked = psms.sort_values("score", ascending=False)
    return ranked.groupby("peptide", sort=False).agg(
        score=("score", "first"),
        is_decoy=("is_decoy", "first"),
        proteins=("proteins", lambda s: sorted({p for ps in s for p in ps})),
    ).reset_index()


def level_stats(df, qcol, fdr):
    """Count targets and decoys passing ``fdr`` and report the empirical FDR."""
    passing = df[df[qcol] <= fdr]
    t = int((~passing["is_decoy"]).sum())
    d = int(passing["is_decoy"].sum())
    return {
        "total": len(df),
        "total_targets": int((~df["is_decoy"]).sum()),
        "total_decoys": int(df["is_decoy"].sum()),
        "passing_targets": t,
        "passing_decoys": d,
        "empirical_fdr": (d / t) if t else None,
    }


def _join(values):
    return ";".join(str(v) for v in values)


def run_pipeline(psm_path, fasta_path, out_dir, fdr=0.01, columns=None,
                 decoy_prefix=DEFAULT_DECOY_PREFIX, taxonomy_path=None,
                 pseudocount=1.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    entries = list(read_fasta(fasta_path))
    taxids = {e.accession: e.taxid for e in entries}
    parents = read_taxonomy(taxonomy_path) if taxonomy_path else {}
    db = check_database(entries, decoy_prefix)
    warnings = list(db["warnings"])

    # 2. PSM level
    psms = load_psms(psm_path, columns=columns, decoy_prefix=decoy_prefix)
    psms["q_psm"] = qvalues(psms["score"], psms["is_decoy"], pseudocount=pseudocount)

    # 3. Peptide level
    peptides = aggregate_peptides(psms)
    peptides["q_peptide"] = qvalues(peptides["score"], peptides["is_decoy"],
                                    pseudocount=pseudocount)
    evidence = peptides[peptides["q_peptide"] <= fdr]

    # 4. Protein level: each protein scored by its best passing peptide
    prot_scores = {}
    for score, accs in zip(evidence["score"], evidence["proteins"]):
        for a in accs:
            if score > prot_scores.get(a, -math.inf):
                prot_scores[a] = score
    proteins = picked_protein_fdr(prot_scores, decoy_prefix, pseudocount=pseudocount)
    accepted = proteins[(~proteins["is_decoy"]) & (proteins["q_protein"] <= fdr)]
    acc_scores = dict(zip(accepted["accession"], accepted["score"]))

    # 5. Protein groups and taxonomy
    target_peps = evidence[~evidence["is_decoy"]].copy()
    target_peps["accepted_proteins"] = [
        [p for p in ps if p in acc_scores] for ps in target_peps["proteins"]]
    keep = np.array([len(ps) > 0 for ps in target_peps["accepted_proteins"]], dtype=bool)
    target_peps = target_peps[keep].copy()
    target_peps["taxid"] = [
        lca([taxids.get(p) for p in ps], parents) for ps in target_peps["accepted_proteins"]]

    groups = greedy_groups(
        dict(zip(target_peps["peptide"], target_peps["accepted_proteins"])), acc_scores)
    qmap = dict(zip(accepted["accession"], accepted["q_protein"]))
    for g in groups:
        g["taxid"] = taxids.get(g["accession"])
        g["q_protein"] = qmap[g["accession"]]

    # Summaries
    psm_pass = psms[(psms["q_psm"] <= fdr) & (~psms["is_decoy"])]
    summary = {
        "fdr_threshold": fdr,
        "database": db,
        "psm": level_stats(psms.rename(columns={"q_psm": "q"}), "q", fdr),
        "peptide": level_stats(peptides.rename(columns={"q_peptide": "q"}), "q", fdr),
        "protein": level_stats(proteins.rename(columns={"q_protein": "q"}), "q", fdr),
        "protein_groups": len(groups),
        "warnings": warnings,
    }
    if summary["protein"]["passing_decoys"] > 0:
        warnings.append("decoy proteins pass the protein FDR; inspect the "
                        "decoy-to-target ratio before trusting protein groups")

    # Outputs
    psm_pass.to_csv(out / "psms_target.tsv", sep="\t", index=False,
                    columns=["spectrum", "peptide", "score", "q_psm"])
    peptides.assign(proteins=peptides["proteins"].map(_join)).to_csv(
        out / "peptides_all.tsv", sep="\t", index=False)
    proteins.to_csv(out / "proteins_all.tsv", sep="\t", index=False)

    pd.DataFrame(groups).to_csv(out / "protein_groups.tsv", sep="\t", index=False)

    tax = target_peps.groupby("taxid", dropna=False).agg(
        peptides=("peptide", "nunique")).reset_index()
    tax.to_csv(out / "taxonomy_peptides.tsv", sep="\t", index=False)

    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Metaproteomics target-decoy FDR pipeline")
    p.add_argument("--psm", required=True, help="PSM table (TSV)")
    p.add_argument("--fasta", required=True, help="target+decoy FASTA used for the search")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--fdr", type=float, default=0.01)
    p.add_argument("--decoy-prefix", default=DEFAULT_DECOY_PREFIX)
    p.add_argument("--taxonomy", help="child/parent TSV with columns taxid, parent")
    p.add_argument("--col-peptide", default="peptide")
    p.add_argument("--col-proteins", default="proteins")
    p.add_argument("--col-score", default="score")
    p.add_argument("--col-spectrum", default="spectrum")
    args = p.parse_args(argv)

    columns = {"peptide": args.col_peptide, "proteins": args.col_proteins,
               "score": args.col_score, "spectrum": args.col_spectrum}
    summary = run_pipeline(args.psm, args.fasta, args.out, fdr=args.fdr,
                           columns=columns, decoy_prefix=args.decoy_prefix,
                           taxonomy_path=args.taxonomy)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
