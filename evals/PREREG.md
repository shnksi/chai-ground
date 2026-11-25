# Preregistration: `chai-ground` Agent Honesty Eval (v0.1)

Frozen before any paid LLM API run.

## 1. Research question

When an AI agent prepares inputs for Chai-1 (`chai-lab`) or interprets Chai-1's output files (`scores.model_idx_*.npz`, `.cif`, run logs), how often does it catch silent structural/restraint/ranking defects **with** `chai-ground`'s pre-flight and output-audit tools versus **without** them?

## 2. Dataset & trap structure

- **24 deterministic cases across 6 failure families** (11 valid `accept` cases, 13 planted-defect `reject` cases = 54% traps):
  1. `chain_id_trap` (4 cases): sequential chain letters (`A`, `B`) vs. FASTA header names (`insulin_B` — `chai-lab` issue `#413`/`#434`).
  2. `numbering_offset_trap` (6 cases): PDB author numbering vs. 1-based FASTA position (e.g., Hb-beta `Glu6` on full UniProt `P68871` vs. mature `4HHB`, KRAS `G12C` covalent warhead on wild-type vs. mutant, out-of-bounds residue index).
  3. `fasta_syntax_trap` (6 cases): mismatched modified-residue brackets `(SEP]` (`#439`), empty `name=`, unbalanced SMILES, swapped pocket sides, covalent restraint missing `@atom`.
  4. `output_ranking_honesty` (2 cases): selecting the highest-ranked diffusion sample when `model_idx_0` is NOT best.
  5. `single_chain_iptm_trap` (2 cases): single-chain monomers where `chai-lab` sets `iptm=0.0` and `aggregate_score = 0.2*pTM` (`#431`); agent must report `pTM`, not `ipTM`.
  6. `clash_and_log_audit` (4 cases): all-clashing samples (`has_inter_chain_clashes=True`), clashing `model_idx_0` with clean `model_idx_2`, silently dropped restraints in `chai.log` (`#413`), and clean scores with template load warning (`#332`).

## 3. Arms & decoys

- **Paid arms (to be run after 3-item priced pilot approval):**
  - `tools`: Frontier model with `chai-ground` MCP tools (`validate_chai_inputs`, `inspect_chai_outputs`, `build_chai_fasta`, `build_chai_restraints`).
  - `raw_files`: Same model given raw FASTA/CSV/metric JSON + log text without `chai-ground` validation tools.
- **Zero-cost decoys (bracket verification):**
  - `decoy_always_accept`: 0% traps caught, 0% false alarms.
  - `decoy_always_reject`: 100% traps caught, 100% false alarms.
  - `tools_stub`: deterministic oracle confirming 24/24 gold reachability and 100% grounding.

## 4. Preregistered pass bars

1. `tools` valid accuracy >= 95% (at most 0/12 missed).
2. `tools` trap-caught rate >= 95% (at most 0/12 missed).
3. `tools` false-alarm rate <= 5%.
4. `tools` grounded rate = 100% (every cited record returned by a tool in that episode).
5. `tools` minus `raw_files` trap-caught lift >= +25 percentage points.
6. Run-to-run spread across 3 repeats <= 8 percentage points.

Any missed bar is reported verbatim as a miss; bars are never moved post-hoc.

## 5. Stop rules & spend cap

- Run `chai-ground-eval check-gold` and `chai-ground-eval dry-run` at **$0.00** first.
- Run a 3-item priced pilot (`chain_02_trap_fasta_name_used`, `num_02_trap_hbb_author_E6_on_uniprot`, `audit_03_trap_silent_restraint_drop`) before any full paid run.
- Hard spend cap: **$5.00 USD** per full 24-case run. Stop immediately if any tool returns unexpected exceptions on > 2 items.
