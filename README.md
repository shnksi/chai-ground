# chai-ground

Pre-flight input verification, restraint grounding, and output-honesty auditing for **Chai-1 (`chaidiscovery/chai-lab`)**, shipped as an MCP server with a preregistered agent failure-mode eval.

> *"A great PRD and a great eval are the same artifact."*

## Why this exists

When an AI agent drives a structure-prediction tool like `chai-lab`, the dangerous failures aren't crashes—they are **runs that finish with exit code 0 and plausible numbers while silently doing the wrong thing**:

1. **Silent restraint drop on FASTA entity names (`chai-lab` #413 / #434):** An agent writes `sequence_1` instead of sequential chain letter `A` in `.restraints`. `chai-lab` catches the tensor-padding error, logs it, and runs **unconstrained**.
2. **Author vs. 1-based FASTA residue drift (#435):** Papers and PDB files use author numbering (e.g., sickle-cell haemoglobin beta `Glu6` is position **7** in full UniProt `P68871`, while position 6 is `Pro`). Passing `E6` on the UniProt sequence silently mis-places or drops the restraint.
3. **Malformed modified-residue brackets & SMILES (#439):** `(SEP]` or truncated SMILES strings break parsing or corrupt conformers.
4. **Single-chain `ipTM` distortion (#431):** For monomers, `chai-lab` computes `aggregate_score = 0.2 * pTM + 0.8 * 0.0`, making a high-confidence `pTM = 0.92` monomer look like `0.184` unless you know to read `pTM` and suppress `ipTM`.
5. **Steric-clash & sample-index traps:** Reporting `model_idx_0` when `model_idx_3` ranked higher, or reporting `pTM = 0.91` on a sample whose `has_inter_chain_clashes` flag is `True`.

`chai-ground` puts deterministic checks at both boundaries—**before** you burn GPU dollars and **after** `chai-lab` writes `.npz` + `.cif` + logs—and pairs them with a 24-case preregistered eval (12 valid specs, 12 planted traps, deterministic graders, zero LLM-as-a-judge).

## The MCP tools (`src/chai_ground/tools.py`)

Every tool returns a typed `Cited[T]` object (`found`, `data`, `note`, `warnings`, `sources`):

| Tool | What it does |
|---|---|
| `build_chai_fasta` | Validates entity types (`protein`, `ligand`, `dna`, `rna`, `glycan`), modified-residue brackets, and SMILES; returns sequential `chai-lab` chain IDs (`A`, `B`, `C`...) with UniProt/PDB provenance. |
| `build_chai_restraints` | Builds a `.restraints` CSV, resolves entity names to sequential chain IDs with explicit warnings, and verifies 1-based residue positions and exact wild-type residue letters. |
| `validate_chai_inputs` | Pre-flight validator catching chain-ID bugs, residue mismatches, out-of-bounds indices, invalid SMILES, swapped pocket sides, and missing covalent `@atom` targets. |
| `inspect_chai_outputs` | Audits a `chai-lab` output directory (`scores.model_idx_*.npz` matching `chai_lab.ranking.rank.get_scores`, `.cif` presence, and `chai.log`)—ranks samples properly, suppresses single-chain `ipTM`, flags steric clashes, and flags dropped restraints in logs. |

## Quickstart

```bash
uv sync
uv run pytest
uv run mypy
uv run ruff check .

# Verify all 24 preregistered gold cases and decoy brackets ($0, offline)
uv run chai-ground-eval check-gold
uv run chai-ground-eval dry-run

# Run as an MCP server over stdio
uv run chai-ground-mcp
```

## Optional: real GPU fold on Modal (~$0.05/complex on L40S)

[`scripts/modal_fold.py`](scripts/modal_fold.py) runs `validate_chai_inputs` locally before launching a GPU, runs `chai-lab fold` remotely capturing full stdout/stderr to `chai.log`, pulls down `.npz`/`.cif`/`chai.log`, and runs `inspect_chai_outputs` locally.

## Preregistered eval

See [`evals/PREREG.md`](evals/PREREG.md) for the frozen bars, arms, decoys, and spend caps, and [`evals/dry-run/DRY_RUN_REPORT.md`](evals/dry-run/DRY_RUN_REPORT.md) for the zero-cost verification output.

MIT License. Sibling project to [proteinground](https://github.com/shnksi/protein-ground).
