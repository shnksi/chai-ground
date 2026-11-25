# chai-ground

Pre-flight input verification, restraint grounding, and output-honesty auditing for **Chai-1 (`chaidiscovery/chai-lab`)**, shipped as an MCP server with a preregistered agent failure-mode eval.

```text
> Validate a Chai-1 restraint putting haemoglobin beta Glu6 on full UniProt P68871

validate_chai_inputs(fasta_text=..., restraints_csv="r1,hbb,E6,ligand,,contact,1.0,0.0,6.0,")
→ valid: false
  issues:
    - [CHAIN_ID_NOT_SEQUENTIAL] Unknown chain 'hbb' — 'hbb' is a FASTA entity name,
      not a chai-lab chain letter; use 'A' instead
    - [RESIDUE_LETTER_MISMATCH] Chain A (hbb) position 6 is 'P', not 'E'.
      Check for PDB-author vs 1-based FASTA numbering offset.

> Audit a single-chain monomer fold directory

inspect_chai_outputs(output_dir="chai_out/")
→ best_model_idx: 2   model_idx_0_is_best: false   trustworthy: true
  single_chain_iptm_caveat: "Single-chain monomer: ipTM is not applicable
    (Chai-1 sets aggregate_score = 0.2*pTM + 0.8*0). Report pTM rather than ipTM."
```

## Why this exists

When an AI agent drives a structure-prediction tool like `chai-lab`, the dangerous failures aren't crashes — they are **runs that finish with exit code 0 and plausible numbers while silently doing the wrong thing**:

1. **Silent restraint drop on FASTA entity names (`chai-lab` #413 / #434):** An agent writes `sequence_1` instead of sequential chain letter `A` in `.restraints`. `chai-lab` catches the error, logs it, and runs **unconstrained**.
2. **Author vs. 1-based FASTA residue drift (#435):** Papers and PDB files use author numbering (e.g., sickle-cell haemoglobin beta `Glu6` is position **7** in full UniProt `P68871`, while position 6 is `Pro`). Passing `E6` on the UniProt sequence misplaces or silently drops the restraint.
3. **Malformed modified-residue brackets & SMILES (#439):** `(SEP]` or unbalanced SMILES strings break parsing or corrupt conformers.
4. **Single-chain `ipTM` distortion (#431):** For monomers, `chai-lab` computes `aggregate_score = 0.2 * pTM + 0.8 * 0.0`, making a high-confidence `pTM = 0.92` monomer look like `0.184` unless you rank by `pTM` and suppress `ipTM`.
5. **Steric-clash, sample-index & log traps:** Reporting `model_idx_0` when `model_idx_3` ranked higher, reporting high `pTM` on a sample whose `has_inter_chain_clashes` flag is `True`, or claiming a fold was restraint-guided when `chai.log` logged dropped restraints or failed template loads.

`chai-ground` puts deterministic checks at both boundaries — **before** you burn GPU dollars and **after** `chai-lab` writes `.npz` + `.cif` + logs — and pairs them with a 24-case preregistered eval (11 valid specs, 13 planted traps, deterministic graders, no LLM-as-a-judge).

## The MCP tools

Defined once in [`src/chai_ground/tools.py`](src/chai_ground/tools.py) and exposed read-only over stdio by [`src/chai_ground/server.py`](src/chai_ground/server.py). Every tool returns a typed `Cited[T]` object (`found`, `data`, `note`, `warnings`, `sources`):

| Tool | What it does |
|---|---|
| `build_chai_fasta` | Validates entity types (`protein`, `ligand`, `dna`, `rna`, `glycan`), modified-residue brackets, and SMILES syntax; returns sequential `chai-lab` chain IDs (`A`, `B`, `C`...) with optional UniProt/PDB provenance. |
| `build_chai_restraints` | Builds a `.restraints` CSV, resolves FASTA entity names to sequential chain letters (`A`, `B`, ...) with explicit warnings, and verifies 1-based residue positions and exact wild-type residue letters. |
| `validate_chai_inputs` | Pre-flight validator catching non-sequential chain IDs, residue letter mismatches, out-of-bounds indices, invalid SMILES, malformed brackets, swapped pocket sides, invalid distance/confidence bounds, and missing covalent `@atom` targets. |
| `inspect_chai_outputs` | Audits a `chai-lab` output directory (`scores.model_idx_*.npz` matching `chai_lab.ranking.rank.get_scores`, `pred.model_idx_*.cif` presence, and `chai.log` / `run.log` / `stderr.log`) — ranks samples properly, suppresses single-chain `ipTM`, flags inter-chain steric clashes, and flags dropped restraints or template load errors in logs. |

## Install

Add it to Claude Code:

```bash
claude mcp add chai-ground -- uvx --from git+https://github.com/shnksi/chai-ground chai-ground-mcp
```

Or to an MCP client config:

```json
{
  "mcpServers": {
    "chai-ground": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/shnksi/chai-ground", "chai-ground-mcp"]
    }
  }
}
```

For local development:

```bash
git clone https://github.com/shnksi/chai-ground && cd chai-ground
uv sync
uv run chai-ground-mcp          # MCP over stdio
```

Pair it with [proteinground](https://github.com/shnksi/protein-ground) (`uvx proteinground`) to fetch canonical UniProt sequences and translate PDB author residue numbering into 1-based sequence coordinates before passing specs to `chai-ground`.

## Optional: real GPU fold on Modal (~$0.05/complex on L40S)

[`scripts/modal_fold.py`](scripts/modal_fold.py) runs `validate_chai_inputs` locally before spinning up a GPU, runs `chai-lab fold` remotely on an L40S (`chai_lab==0.6.1`) capturing full stdout/stderr to `chai.log`, pulls down `.npz`/`.cif`/`chai.log`, and runs `inspect_chai_outputs` locally:

```bash
pip install modal && modal setup
modal run scripts/modal_fold.py --fasta path/to/complex.fasta --restraints path/to/complex.restraints
```

## Preregistered eval

24 deterministic cases across 6 failure families (**11 valid specs, 13 planted traps**): `chain_id_trap` (4), `numbering_offset_trap` (6), `fasta_syntax_trap` (6), `output_ranking_honesty` (2), `single_chain_iptm_trap` (2), and `clash_and_log_audit` (4).

Bars, arms, decoys, and spend caps are frozen in [`evals/PREREG.md`](evals/PREREG.md). Zero-cost offline verification (`evals/dry-run/DRY_RUN_REPORT.md`):

| Arm | Valid correct (11) | Traps caught (13) | False alarms | Overall | Grounded | Cost |
|---|---:|---:|---:|---:|---:|---:|
| `tools_stub` | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | $0.00 |
| `decoy_always_accept` | 54.5% | 0.0% | 0.0% | 25.0% | 0.0% | $0.00 |
| `decoy_always_reject` | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | $0.00 |

Paid LLM arms (`tools` vs `raw_files`) are preregistered in `evals/PREREG.md` and have not yet been run; the CLI currently implements the offline verification commands (`check-gold` and `dry-run`).

## Development

```bash
uv sync
uv run pytest                   # 9 tests, offline
uv run mypy                     # strict
uv run ruff check .
uv run chai-ground-eval check-gold
uv run chai-ground-eval dry-run
```

## Limits

- **Syntactic SMILES checking only:** `validate_smiles` checks character set, balanced parentheses/brackets, and atom symbols without an RDKit dependency; it will not catch chemically invalid valences or unkekulizable rings.
- **No direct database lookups:** `chai-ground` validates FASTA and restraints against each other offline. Use [proteinground](https://github.com/shnksi/protein-ground) to retrieve canonical sequences and verify PDB-to-UniProt numbering offsets first.
- **Log auditing requires captured logs:** Silent restraint drops and template errors are detected by scanning `chai.log` (or `run.log` / `stderr.log`, or an explicit `log_path`) against known `chai-lab` log messages; runs whose stderr was discarded cannot be audited for log-only drops.
- **Eval status:** The 24 offline cases, deterministic graders, and decoy brackets are verified at $0; a live LLM comparison (`tools` vs `raw_files`) has not yet been run.

MIT License. Sibling project to [proteinground](https://github.com/shnksi/protein-ground).
