# Proteomics

## Metaproteomics FDR pipeline

Target-decoy FDR control for large, multi-species databases, with separate
control at PSM, peptide and protein level.

### Why FDR is harder with a large database

- A concatenated target+decoy FASTA must have a decoy for every target. A
  mismatch makes the estimate unreliable, and the pipeline warns about it.
- Protein inference done by counting every decoy hit inflates the protein FDR
  when many proteins share peptides. The pipeline uses **picked protein FDR**:
  each target is paired with its decoy, and only the better of the pair is kept.
- The estimator uses `(decoys + 1) / targets`. With few targets, 1% FDR needs
  roughly 100 or more of them before anything passes.

### Usage

```bash
pip install -r requirements.txt

# 1. Build the target+decoy database used for the search
python -c "from metaproteomics import write_decoy_fasta as w; w('db.fasta', 'db_td.fasta')"

# 2. Run the FDR pipeline on the search output
python -m metaproteomics --psm psm.tsv --fasta db_td.fasta --out results \
    --fdr 0.01 --col-peptide Peptide --col-proteins Protein --col-score Hyperscore \
    --taxonomy taxonomy.tsv   # optional: columns taxid, parent
```

The PSM table is tab-separated. Protein accessions must match the FASTA header
accession (the first token after `>`). Modifications are stripped from peptide
sequences, so `M[+15.99]` and `M` count as the same peptide.

### Outputs (in `--out`)

| File | Contents |
|---|---|
| `psms_target.tsv` | target PSMs passing the PSM-level threshold |
| `peptides_all.tsv` | all peptides with q-values and decoy flags |
| `proteins_all.tsv` | protein winners of the picked competition with q-values |
| `protein_groups.tsv` | representative proteins, unique and shared peptide counts, taxid |
| `taxonomy_peptides.tsv` | peptide counts per LCA taxid |
| `summary.json` | per-level target and decoy counts, empirical FDR, warnings |

### Tests

```bash
python -m pytest -q tests
```
