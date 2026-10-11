"""Readers for FASTA databases and PSM tables, and decoy generation."""
import re
from collections import namedtuple

import pandas as pd

DEFAULT_DECOY_PREFIX = "rev_"
_OX = re.compile(r"\bOX=(\d+)")

FastaEntry = namedtuple("FastaEntry", "accession taxid sequence header")


def clean_sequence(seq):
    """Keep only the amino-acid letters, dropping modifications and flanks."""
    return re.sub(r"[^A-Z]", "", str(seq).upper())


def read_fasta(path):
    """Yield a FastaEntry per record.

    The accession is the first whitespace-delimited token of the header, and
    the taxid comes from an ``OX=`` field (UniProt convention) when present.
    """
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield _entry(header, chunks)
                header, chunks = line[1:], []
            elif line:
                chunks.append(line)
    if header is not None:
        yield _entry(header, chunks)


def _entry(header, chunks):
    m = _OX.search(header)
    return FastaEntry(
        accession=header.split()[0],
        taxid=int(m.group(1)) if m else None,
        sequence="".join(chunks),
        header=header,
    )


def write_decoy_fasta(in_path, out_path, prefix=DEFAULT_DECOY_PREFIX):
    """Write the targets followed by reversed-sequence decoys.

    Decoy accessions are the target accession with ``prefix`` prepended, so
    the pipeline can pair each decoy with its target.
    """
    with open(out_path, "w") as out:
        entries = list(read_fasta(in_path))
        for e in entries:
            _write_record(out, e.header, e.sequence)
        for e in entries:
            _write_record(out, prefix + e.header, e.sequence[::-1])


def _write_record(out, header, sequence, width=60):
    out.write(f">{header}\n")
    for i in range(0, len(sequence), width):
        out.write(sequence[i:i + width] + "\n")


def load_psms(path, columns=None, decoy_prefix=DEFAULT_DECOY_PREFIX, sep="\t"):
    """Load a PSM table into a DataFrame.

    Output columns: spectrum, peptide (residues only), proteins (list),
    score (higher is better) and is_decoy. A PSM is a decoy only when every
    protein it maps to carries the decoy prefix, so a PSM that also maps to a
    target protein counts as a target.

    ``columns`` maps the required fields onto the file's own headers, e.g.
    ``{"peptide": "Peptide", "proteins": "Protein", "score": "Hyperscore"}``.
    """
    cols = {"spectrum": "spectrum", "peptide": "peptide",
            "proteins": "proteins", "score": "score", **(columns or {})}
    raw = pd.read_csv(path, sep=sep, dtype=str)
    missing = [v for v in cols.values() if v not in raw.columns]
    if missing:
        raise ValueError(f"PSM file is missing columns: {missing}")

    df = pd.DataFrame({
        "spectrum": raw[cols["spectrum"]],
        "peptide": raw[cols["peptide"]].fillna("").map(clean_sequence),
        "proteins": raw[cols["proteins"]].fillna("").map(
            lambda s: [p.strip() for p in s.split(";") if p.strip()]),
        "score": pd.to_numeric(raw[cols["score"]], errors="coerce"),
    })
    df = df[(df["peptide"] != "") & df["proteins"].map(bool) & df["score"].notna()]
    df = df.reset_index(drop=True)
    df["is_decoy"] = df["proteins"].map(
        lambda ps: all(p.startswith(decoy_prefix) for p in ps))
    return df
