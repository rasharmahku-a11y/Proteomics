import json
import random

import numpy as np
import pytest

from metaproteomics import (
    greedy_groups, lca, load_psms, picked_protein_fdr, qvalues,
    read_fasta, read_taxonomy, run_pipeline, write_decoy_fasta,
)


# ---- FDR -------------------------------------------------------------------

def test_qvalues_hand_computed():
    # Ranked: 10(T) 9(D) 8(T) 7(T); pseudocount 0
    # FDR per rank: 0/1, 1/1, 1/2, 1/3 -> q (min over lower ranks): 0, 1/3, 1/3, 1/3
    q = qvalues([10, 9, 8, 7], [False, True, False, False], pseudocount=0)
    assert np.allclose(q, [0, 1 / 3, 1 / 3, 1 / 3])


def test_qvalues_ties_share_threshold():
    # 5(T) and 5(D) tie; the tie group is judged together.
    # Group {5,5}: targets 1, decoys 1 -> FDR 1. Group {4}: targets 2, decoys 1 -> 0.5.
    # q for the 5-group = min(1, 0.5) = 0.5.
    q = qvalues([5, 5, 4], [False, True, False], pseudocount=0)
    assert np.allclose(q, [0.5, 0.5, 0.5])


def test_qvalues_lower_is_better():
    q_low = qvalues([1, 2, 3], [False, True, False], higher_better=False, pseudocount=0)
    q_high = qvalues([-1, -2, -3], [False, True, False], higher_better=True, pseudocount=0)
    assert np.allclose(q_low, q_high)


def test_qvalues_monotone_and_bounded():
    rng = random.Random(0)
    scores = [rng.random() for _ in range(500)]
    decoy = [rng.random() < 0.5 for _ in range(500)]
    q = qvalues(scores, decoy)
    assert (q >= 0).all() and (q <= 1).all()
    order = np.argsort(-np.asarray(scores))
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_qvalues_rejects_nan():
    with pytest.raises(ValueError):
        qvalues([1.0, np.nan], [False, True])


# ---- Picked protein FDR ----------------------------------------------------

def test_picked_protein_fdr_keeps_better_of_pair():
    scores = {"P1": 10.0, "rev_P1": 5.0,   # target wins
              "P2": 3.0,                   # unpaired target
              "rev_P3": 8.0}               # unpaired decoy
    df = picked_protein_fdr(scores, "rev_", pseudocount=0)
    winners = dict(zip(df.accession, df.is_decoy))
    assert winners == {"P1": False, "P2": False, "rev_P3": True}


def test_picked_protein_fdr_drops_ties():
    df = picked_protein_fdr({"P1": 4.0, "rev_P1": 4.0}, "rev_", pseudocount=0)
    assert df.empty


# ---- Protein groups --------------------------------------------------------

def test_greedy_groups_keeps_protein_with_unique_peptide():
    peptide_proteins = {"pepA": ["P1", "P2"], "pepB": ["P1"], "pepC": ["P2"]}
    groups = greedy_groups(peptide_proteins, {"P1": 10.0, "P2": 5.0})
    assert {g["accession"] for g in groups} == {"P1", "P2"}
    p2 = next(g for g in groups if g["accession"] == "P2")
    assert p2["unique_peptides"] == 1 and p2["shared_peptides"] == 1


def test_greedy_groups_removes_fully_covered_protein():
    peptide_proteins = {"pepA": ["P1", "P2"], "pepB": ["P1"]}
    groups = greedy_groups(peptide_proteins, {"P1": 10.0, "P2": 5.0})
    assert [g["accession"] for g in groups] == ["P1"]


# ---- Taxonomy --------------------------------------------------------------

PARENTS = {9606: 9605, 9598: 9605, 9605: 1, 10090: 1, 1: None}


def test_lca_shared_genus():
    assert lca([9606, 9598], PARENTS) == 9605


def test_lca_across_kingdoms_is_root():
    assert lca([9606, 10090], PARENTS) == 1


def test_lca_single_and_unknown():
    assert lca([9606, 9606], PARENTS) == 9606
    assert lca([9606, None], PARENTS) == 9606
    assert lca([None], PARENTS) is None
    assert lca([], PARENTS) is None


def test_lca_unknown_taxa_without_taxonomy():
    assert lca([9606, 10090], {}) is None


# ---- I/O -------------------------------------------------------------------

def _write_fasta(path, records):
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n{seq}\n")


def test_fasta_parsing_and_decoys(tmp_path):
    src = tmp_path / "db.fasta"
    _write_fasta(src, [("sp|P1|A_HUMAN desc OX=9606", "ACDEFGK"),
                       ("sp|P2|B_MOUSE desc OX=10090", "MKLLPR")])
    entries = list(read_fasta(src))
    assert [e.taxid for e in entries] == [9606, 10090]
    assert entries[0].accession == "sp|P1|A_HUMAN"

    out = tmp_path / "td.fasta"
    write_decoy_fasta(src, out)
    combined = list(read_fasta(out))
    assert len(combined) == 4
    decoys = [e for e in combined if e.accession.startswith("rev_")]
    assert decoys[0].sequence == "ACDEFGK"[::-1]


def test_load_psms_decoy_flag(tmp_path):
    psm = tmp_path / "psm.tsv"
    psm.write_text(
        "spectrum\tpeptide\tproteins\tscore\n"
        "s1\tPEPTIDEK\tP1;P2\t50\n"
        "s2\tDECOYK\trev_P1\t20\n"
        "s3\tMIXK\tP1;rev_P9\t30\n"
        "s4\t\tP1\t40\n")  # empty peptide is dropped
    df = load_psms(psm)
    assert len(df) == 3
    assert list(df["is_decoy"]) == [False, True, False]


# ---- End to end ------------------------------------------------------------

def _letters(i):
    """Unique letters-only token for index i (peptide names may not contain digits)."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(ord("A") + r) + out
    return out


def _synthetic_run(tmp_path, n_targets=200, n_decoys=200):
    """Targets score high and decoys low, so FDR at 1% should keep only targets."""
    rng = random.Random(1)
    records = []
    for i in range(n_targets):
        records.append((f"sp|T{i}|X OX={9606 if i % 2 else 10090}", "ACDEFGHIKLMNPQR" * 2))
    fasta = tmp_path / "db.fasta"
    _write_fasta(fasta, records)
    td = tmp_path / "td.fasta"
    write_decoy_fasta(fasta, td)

    lines = ["spectrum\tpeptide\tproteins\tscore"]
    for i in range(n_targets):
        for k in range(3):
            pep = f"PEP{_letters(i)}{_letters(k)}K"
            lines.append(f"s{i}_{k}\t{pep}\tsp|T{i}|X\t{100 + rng.random():.3f}")
    for i in range(n_decoys):
        pep = f"DEC{_letters(i)}K"
        lines.append(f"d{i}\t{pep}\trev_sp|T{i % n_targets}|X\t{rng.random():.3f}")
    psm = tmp_path / "psm.tsv"
    psm.write_text("\n".join(lines) + "\n")
    return psm, td


def test_end_to_end_keeps_only_targets_at_1pct(tmp_path):
    psm, td = _synthetic_run(tmp_path)
    out = tmp_path / "out"
    summary = run_pipeline(psm, td, out, fdr=0.01, pseudocount=1.0)

    n = 200
    assert summary["database"]["targets"] == n
    assert summary["database"]["decoys"] == n
    assert summary["database"]["warnings"] == []
    # Decoys may pass at the peptide level, because the estimator allows
    # (decoys + 1) / targets <= fdr. The empirical FDR must still respect the threshold.
    assert summary["peptide"]["passing_targets"] == 3 * n
    assert summary["peptide"]["empirical_fdr"] <= 0.01
    assert summary["protein"]["passing_decoys"] == 0
    assert summary["protein_groups"] == n

    saved = json.loads((out / "summary.json").read_text())
    assert saved["fdr_threshold"] == 0.01
    assert (out / "protein_groups.tsv").exists()
    assert (out / "taxonomy_peptides.tsv").exists()


def test_database_warning_when_decoys_missing(tmp_path):
    fasta = tmp_path / "targets_only.fasta"
    _write_fasta(fasta, [("sp|T1|X OX=9606", "ACDEK")])
    psm = tmp_path / "psm.tsv"
    psm.write_text("spectrum\tpeptide\tproteins\tscore\ns1\tACDEK\tsp|T1|X\t9\n")
    summary = run_pipeline(psm, fasta, tmp_path / "out", fdr=0.01)
    assert any("no decoy entries" in w for w in summary["database"]["warnings"])
