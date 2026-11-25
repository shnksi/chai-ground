"""Pre-flight FASTA and restraint builders + validators grounded in chai-lab's actual rules.

Prevents silent chai-lab agent failures:
- Issue #413 / #434: using FASTA header names ('sequence_1') instead of sequential chain letters ('A', 'B')
- Issue #435: restraints silently dropped when residue letter or index does not match the chain
- Issue #439: mismatched modified-residue brackets like '(SEP]' or empty FASTA labels
- Author-vs-FASTA numbering drift (e.g., PDB 4HHB Hb-beta Glu6 is 1-based FASTA position 7 in P68871)
"""

from __future__ import annotations

import csv
import io
import re
import string
from datetime import UTC, datetime
from typing import Any

from chai_ground.models import (
    BuiltFasta,
    BuiltRestraints,
    ChainMapping,
    Cited,
    EntityKind,
    RestraintCheck,
    RestraintKind,
    Source,
    ValidationIssue,
    ValidationReport,
)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWYX")
DNA_BASES = set("ACGT")
RNA_BASES = set("ACGU")
VALID_ENTITY_KINDS: set[str] = {"protein", "ligand", "dna", "rna", "glycan"}
VALID_CONNECTION_TYPES: set[str] = {"contact", "pocket", "covalent"}

# Exact allowed SMILES alphabet check (balanced parens/brackets, no spaces)
SMILES_CHARS = set(string.ascii_letters + string.digits + ".-+=#$%:/\\[]()<>@*")
RES_IDX_RE = re.compile(r"^([A-Za-z])(\d+)$")
ATOM_NAME_RE = re.compile(r"^[A-Za-z0-9']+$")

AA_3_TO_1: dict[str, str] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
}


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sequential_chain_id(index: int) -> str:
    """Chai-1 assigns chains A, B, ..., Z, AA, AB... in FASTA order."""
    if index < 0:
        raise ValueError("Chain index must be >= 0")
    if index < 26:
        return chr(ord("A") + index)
    first = (index // 26) - 1
    second = index % 26
    return chr(ord("A") + first) + chr(ord("A") + second)


def parse_modified_fasta_tokens(seq: str) -> tuple[list[str] | None, str | None]:
    """Strict mirror of chai_lab.data.parsing.input_validation.constituents_of_modified_fasta,
    also rejecting mismatched brackets like '(SEP]' or '[SEP)' (#439)."""
    s = seq.strip().upper()
    if not s:
        return None, "Empty sequence"
    allowed = set(string.ascii_uppercase + "()[]" + string.digits)
    bad = [ch for ch in s if ch not in allowed]
    if bad:
        return None, f"Disallowed characters in polymer sequence: {sorted(set(bad))}"

    tokens: list[str] = []
    current_mod: str | None = None
    expected_close: str | None = None

    for ch in s:
        if ch in ("(", "["):
            if current_mod is not None:
                return None, "Nested or unclosed modification bracket"
            current_mod = ""
            expected_close = ")" if ch == "(" else "]"
        elif ch in (")", "]"):
            if current_mod is None:
                return None, f"Closing bracket '{ch}' without matching open bracket"
            if ch != expected_close:
                return (
                    None,
                    f"Mismatched modification brackets: expected '{expected_close}', got '{ch}'",
                )
            if len(current_mod) <= 1:
                return None, f"Empty or single-character modification '({current_mod})'"
            tokens.append(current_mod)
            current_mod = None
            expected_close = None
        else:
            if current_mod is not None:
                current_mod += ch
            else:
                if ch not in string.ascii_uppercase:
                    return None, f"Unexpected digit '{ch}' outside modification brackets"
                tokens.append(ch)

    if current_mod is not None:
        return None, f"Unclosed modification bracket (expected '{expected_close}')"
    return tokens, None


def validate_smiles(smiles: str) -> str | None:
    """Lightweight, deterministic SMILES syntax check (no RDKit dependency required)."""
    s = smiles.strip()
    if not s:
        return "Empty SMILES string"
    if any(ch.isspace() for ch in s):
        return "SMILES contains whitespace"
    bad = [ch for ch in s if ch not in SMILES_CHARS]
    if bad:
        return f"SMILES contains invalid characters: {sorted(set(bad))}"
    paren_depth = 0
    bracket_depth = 0
    for ch in s:
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
            if paren_depth < 0:
                return "Unbalanced parentheses in SMILES"
        elif ch == "[":
            bracket_depth += 1
            if bracket_depth > 1:
                return "Nested square brackets in SMILES"
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth < 0:
                return "Unbalanced square brackets in SMILES"
    if paren_depth != 0 or bracket_depth != 0:
        return "Unclosed parenthesis or bracket in SMILES"
    if not any(ch in string.ascii_letters for ch in s):
        return "SMILES contains no atom symbols"
    return None


def parse_chai_fasta(fasta_text: str) -> tuple[list[ChainMapping], list[ValidationIssue]]:
    """Parse and validate a chai-lab FASTA string into sequential chains (A, B, C, ...)."""
    issues: list[ValidationIssue] = []
    chains: list[ChainMapping] = []

    raw_records: list[tuple[str, str]] = []
    current_header: str | None = None
    seq_parts: list[str] = []

    for line_no, raw_line in enumerate(fasta_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                raw_records.append((current_header, "".join(seq_parts)))
            current_header = line[1:].strip()
            seq_parts = []
            if not current_header:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="EMPTY_FASTA_HEADER",
                        message="FASTA header is empty",
                        location=f"fasta:line {line_no}",
                    )
                )
        else:
            if current_header is None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="SEQUENCE_BEFORE_HEADER",
                        message="Sequence line appeared before any '>' FASTA header",
                        location=f"fasta:line {line_no}",
                    )
                )
            else:
                seq_parts.append(line)

    if current_header is not None:
        raw_records.append((current_header, "".join(seq_parts)))

    if not raw_records:
        issues.append(
            ValidationIssue(
                severity="error",
                code="EMPTY_FASTA",
                message="FASTA input contains no sequences",
                location="fasta",
            )
        )
        return chains, issues

    for idx, (header, seq) in enumerate(raw_records):
        chain_id = sequential_chain_id(idx)
        parts = header.split("|")
        if len(parts) < 2 or "=" not in parts[1]:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="MALFORMED_FASTA_HEADER",
                    message=(
                        f"Header '>{header}' must follow '>entity_type|name=entity_name' "
                        "(e.g. '>protein|name=kras')"
                    ),
                    location=f"chain {chain_id}",
                )
            )
            continue

        entity_raw = parts[0].strip().lower()
        name_key, _, name_val = parts[1].partition("=")
        name_val = name_val.strip()

        if entity_raw not in VALID_ENTITY_KINDS:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="UNKNOWN_ENTITY_TYPE",
                    message=(
                        f"Unknown entity type '{entity_raw}' in '>{header}'; "
                        f"expected one of {sorted(VALID_ENTITY_KINDS)}"
                    ),
                    location=f"chain {chain_id}",
                )
            )
            continue

        if name_key.strip().lower() != "name" or not name_val:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="EMPTY_ENTITY_NAME",
                    message=f"Header '>{header}' has an empty or malformed name= field",
                    location=f"chain {chain_id}",
                )
            )
            continue

        entity_kind: EntityKind = entity_raw  # type: ignore[assignment]
        length = len(seq)

        if entity_kind in ("protein", "dna", "rna"):
            tokens, err = parse_modified_fasta_tokens(seq)
            if err is not None or tokens is None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="INVALID_POLYMER_SEQUENCE",
                        message=f"Chain {chain_id} ({name_val}): {err}",
                        location=f"chain {chain_id}",
                    )
                )
                continue
            length = len(tokens)
            single_letters = {t for t in tokens if len(t) == 1}
            if entity_kind == "protein":
                invalid_aa = sorted(single_letters - STANDARD_AA)
                if invalid_aa:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="INVALID_AMINO_ACID",
                            message=(
                                f"Chain {chain_id} ({name_val}) contains non-standard protein "
                                f"letter(s) {invalid_aa} (use modified-residue brackets or standard AA)"
                            ),
                            location=f"chain {chain_id}",
                        )
                    )
            elif entity_kind == "dna" and not single_letters.issubset(DNA_BASES):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="INVALID_DNA_BASE",
                        message=f"Chain {chain_id} ({name_val}) has invalid DNA base(s): {sorted(single_letters - DNA_BASES)}",
                        location=f"chain {chain_id}",
                    )
                )
            elif entity_kind == "rna" and not single_letters.issubset(RNA_BASES):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="INVALID_RNA_BASE",
                        message=f"Chain {chain_id} ({name_val}) has invalid RNA base(s): {sorted(single_letters - RNA_BASES)}",
                        location=f"chain {chain_id}",
                    )
                )
        elif entity_kind == "ligand":
            smiles_err = validate_smiles(seq)
            if smiles_err is not None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="INVALID_SMILES",
                        message=f"Chain {chain_id} ({name_val}): {smiles_err}",
                        location=f"chain {chain_id}",
                    )
                )
                continue

        chains.append(
            ChainMapping(
                chai_chain_id=chain_id,
                entity_type=entity_kind,
                name=name_val,
                sequence=seq,
                length=length,
            )
        )

    return chains, issues


def _parse_res_and_atom(raw: str) -> tuple[str, str, str | None]:
    """Mirror chai_lab.data.parsing.restraints._parse_res_idx."""
    val = raw.strip()
    if not val:
        return "", "", None
    if val.endswith("@"):
        return "", "", f"Trailing '@' with no atom name in '{val}'"
    parts = val.split("@")
    if len(parts) > 2:
        return "", "", f"Multiple '@' separators in '{val}'"
    res = parts[0].strip()
    atom = parts[1].strip() if len(parts) == 2 else ""
    if not res and not atom:
        return "", "", f"Empty residue and atom in '{val}'"
    if atom and not ATOM_NAME_RE.match(atom):
        return "", "", f"Invalid atom name '{atom}' in '{val}'"
    return res, atom, None


def _check_chain_and_residue(
    *,
    chain_id: str,
    res_token: str,
    atom_name: str,
    side: str,
    restraint_id: str,
    chains_by_id: dict[str, ChainMapping],
    chains_by_name: dict[str, str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    loc = f"restraint {restraint_id} ({side})"

    if chain_id not in chains_by_id:
        hint = ""
        if chain_id in chains_by_name:
            hint = (
                f" — '{chain_id}' is a FASTA entity name, not a chai-lab chain letter; "
                f"use '{chains_by_name[chain_id]}' instead (chai-lab silently drops restraints on unknown chains, issue #413)"
            )
        else:
            hint = f"; available chains are {sorted(chains_by_id.keys())}"
        issues.append(
            ValidationIssue(
                severity="error",
                code="CHAIN_ID_NOT_SEQUENTIAL",
                message=f"Unknown chain '{chain_id}'{hint}",
                location=loc,
            )
        )
        return issues

    chain = chains_by_id[chain_id]
    if not res_token:
        return issues

    if chain.entity_type == "ligand":
        # Ligands in covalent bonds use '@C1' without a residue letter+index.
        issues.append(
            ValidationIssue(
                severity="error",
                code="LIGAND_RESIDUE_INDEX_INVALID",
                message=(
                    f"Chain '{chain_id}' is a ligand (SMILES); reference ligand atoms as '@ATOM' "
                    f"rather than residue '{res_token}'"
                ),
                location=loc,
            )
        )
        return issues

    m = RES_IDX_RE.match(res_token)
    if not m:
        issues.append(
            ValidationIssue(
                severity="error",
                code="MALFORMED_RESIDUE_INDEX",
                message=f"Residue token '{res_token}' must be 1-letter code + 1-based integer (e.g. 'G12')",
                location=loc,
            )
        )
        return issues

    letter = m.group(1).upper()
    pos = int(m.group(2))
    if pos < 1 or pos > chain.length:
        issues.append(
            ValidationIssue(
                severity="error",
                code="RESIDUE_OUT_OF_BOUNDS",
                message=(
                    f"Position {pos} is out of bounds for chain {chain_id} ({chain.name}), "
                    f"which has length {chain.length} (1-based)"
                ),
                location=loc,
            )
        )
        return issues

    tokens, _ = parse_modified_fasta_tokens(chain.sequence)
    if tokens is not None:
        actual = tokens[pos - 1]
        expected_1 = actual if len(actual) == 1 else AA_3_TO_1.get(actual, "X")
        if letter != expected_1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="RESIDUE_LETTER_MISMATCH",
                    message=(
                        f"Chain {chain_id} ({chain.name}) position {pos} is '{expected_1}' "
                        f"(token '{actual}'), not '{letter}'. Check for PDB-author vs 1-based FASTA numbering offset."
                    ),
                    location=loc,
                )
            )
    return issues


def validate_chai_inputs(
    fasta_text: str,
    restraints_csv: str | None = None,
) -> Cited[ValidationReport]:
    """Pre-flight validator for chai-lab FASTA + optional restraints CSV."""
    now = _now_iso()
    chains, issues = parse_chai_fasta(fasta_text)
    chains_by_id = {c.chai_chain_id: c for c in chains}
    chains_by_name = {c.name: c.chai_chain_id for c in chains}
    restraint_count = 0

    if restraints_csv is not None and restraints_csv.strip():
        reader = csv.DictReader(io.StringIO(restraints_csv.strip()))
        required_cols = {
            "restraint_id",
            "chainA",
            "res_idxA",
            "chainB",
            "res_idxB",
            "connection_type",
            "confidence",
            "min_distance_angstrom",
            "max_distance_angstrom",
        }
        fieldnames = set(reader.fieldnames or [])
        missing_cols = sorted(required_cols - fieldnames)
        if missing_cols:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="MISSING_RESTRAINT_COLUMNS",
                    message=f"Restraints CSV is missing required column(s): {missing_cols}",
                    location="restraints.csv",
                )
            )
        else:
            seen_ids: set[str] = set()
            for row_idx, row in enumerate(reader, start=2):
                restraint_count += 1
                rid = (row.get("restraint_id") or "").strip() or f"row_{row_idx}"
                if rid in seen_ids:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="DUPLICATE_RESTRAINT_ID",
                            message=f"Duplicate restraint_id '{rid}'",
                            location=f"restraints.csv:line {row_idx}",
                        )
                    )
                seen_ids.add(rid)

                ctype = (row.get("connection_type") or "").strip().lower()
                if ctype not in VALID_CONNECTION_TYPES:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="INVALID_CONNECTION_TYPE",
                            message=f"Invalid connection_type '{ctype}'; expected one of {sorted(VALID_CONNECTION_TYPES)}",
                            location=f"restraint {rid}",
                        )
                    )
                    continue

                chain_a = (row.get("chainA") or "").strip()
                chain_b = (row.get("chainB") or "").strip()
                res_a, atom_a, err_a = _parse_res_and_atom(row.get("res_idxA") or "")
                res_b, atom_b, err_b = _parse_res_and_atom(row.get("res_idxB") or "")
                if err_a:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="MALFORMED_RESIDUE_INDEX",
                            message=err_a,
                            location=f"restraint {rid} (A)",
                        )
                    )
                if err_b:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="MALFORMED_RESIDUE_INDEX",
                            message=err_b,
                            location=f"restraint {rid} (B)",
                        )
                    )

                # Distance & confidence bounds
                try:
                    min_d = (
                        float(row["min_distance_angstrom"])
                        if (row.get("min_distance_angstrom") or "").strip()
                        else 0.0
                    )
                    max_d = (
                        float(row["max_distance_angstrom"])
                        if (row.get("max_distance_angstrom") or "").strip()
                        else 10.0
                    )
                    if min_d < 0 or max_d < 0 or max_d < min_d:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                code="INVALID_DISTANCE_BOUNDS",
                                message=f"Invalid distance bounds min={min_d}, max={max_d} (require 0 <= min <= max)",
                                location=f"restraint {rid}",
                            )
                        )
                except ValueError:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="NON_NUMERIC_DISTANCE",
                            message="min_distance_angstrom / max_distance_angstrom must be numeric",
                            location=f"restraint {rid}",
                        )
                    )

                try:
                    conf = (
                        float(row["confidence"]) if (row.get("confidence") or "").strip() else 1.0
                    )
                    if not (0.0 <= conf <= 1.0):
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                code="INVALID_CONFIDENCE",
                                message=f"confidence={conf} outside [0.0, 1.0]",
                                location=f"restraint {rid}",
                            )
                        )
                except ValueError:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="NON_NUMERIC_CONFIDENCE",
                            message="confidence must be a float in [0.0, 1.0]",
                            location=f"restraint {rid}",
                        )
                    )

                # Type-specific rules from chai_lab.data.parsing.restraints.PairwiseInteraction
                if ctype == "pocket":
                    if res_a or atom_a:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                code="POCKET_SIDE_A_NOT_EMPTY",
                                message="For pocket restraints, res_idxA must be empty (chain-level partner)",
                                location=f"restraint {rid}",
                            )
                        )
                    if not res_b or atom_b:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                code="POCKET_SIDE_B_INVALID",
                                message="For pocket restraints, res_idxB must name a residue with no @atom",
                                location=f"restraint {rid}",
                            )
                        )
                elif ctype == "contact":
                    if not (res_a or atom_a) or not (res_b or atom_b):
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                code="CONTACT_ENDPOINT_EMPTY",
                                message="Contact restraints require both res_idxA and res_idxB",
                                location=f"restraint {rid}",
                            )
                        )
                elif ctype == "covalent" and (not atom_a or not atom_b):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="COVALENT_MISSING_ATOM",
                            message="Covalent restraints must specify @atom on both partners (e.g. 'C12@SG' and '@C1')",
                            location=f"restraint {rid}",
                        )
                    )

                issues.extend(
                    _check_chain_and_residue(
                        chain_id=chain_a,
                        res_token=res_a,
                        atom_name=atom_a,
                        side="A",
                        restraint_id=rid,
                        chains_by_id=chains_by_id,
                        chains_by_name=chains_by_name,
                    )
                )
                issues.extend(
                    _check_chain_and_residue(
                        chain_id=chain_b,
                        res_token=res_b,
                        atom_name=atom_b,
                        side="B",
                        restraint_id=rid,
                        chains_by_id=chains_by_id,
                        chains_by_name=chains_by_name,
                    )
                )

    has_errors = any(i.severity == "error" for i in issues)
    report = ValidationReport(
        valid=not has_errors,
        chains=chains,
        restraint_count=restraint_count,
        issues=issues,
    )
    return Cited[ValidationReport](
        found=True,
        data=report,
        note=None
        if report.valid
        else f"Validation failed with {sum(i.severity == 'error' for i in issues)} error(s)",
        sources=[
            Source(
                kind="chai_lab_spec",
                record_id="chai-lab/data/parsing/{input_validation,restraints}.py",
                url="https://github.com/chaidiscovery/chai-lab",
                retrieved_at=now,
            )
        ],
    )


def build_chai_fasta_from_entities(
    entities: list[dict[str, Any]],
) -> Cited[BuiltFasta]:
    """Construct and pre-flight validate a chai-lab FASTA from structured entity definitions.

    Each entity dict expects:
      - entity_type: 'protein' | 'ligand' | 'dna' | 'rna' | 'glycan'
      - name: str
      - sequence: str
      - optional metadata: uniprot_accession, pdb_id, pdb_chain, numbering_offset_note
    """
    now = _now_iso()
    if not entities:
        return Cited[BuiltFasta](
            found=False,
            data=None,
            note="Cannot build FASTA from an empty entity list",
        )

    lines: list[str] = []
    for ent in entities:
        etype = str(ent.get("entity_type", "")).strip().lower()
        name = str(ent.get("name", "")).strip()
        seq = str(ent.get("sequence", "")).strip()
        lines.append(f">{etype}|name={name}")
        lines.append(seq)
    fasta_text = "\n".join(lines) + "\n"

    val = validate_chai_inputs(fasta_text)
    assert val.data is not None
    if not val.data.valid:
        errs = "; ".join(i.message for i in val.data.issues if i.severity == "error")
        return Cited[BuiltFasta](
            found=False,
            data=None,
            note=f"Invalid Chai-1 FASTA specification: {errs}",
            sources=val.sources,
        )

    enriched: list[ChainMapping] = []
    sources: list[Source] = list(val.sources)
    for chain, ent in zip(val.data.chains, entities, strict=True):
        acc = ent.get("uniprot_accession")
        pdb_id = ent.get("pdb_id")
        pdb_chain = ent.get("pdb_chain")
        offset_note = ent.get("numbering_offset_note")
        enriched.append(
            chain.model_copy(
                update={
                    "uniprot_accession": acc,
                    "pdb_id": pdb_id,
                    "pdb_chain": pdb_chain,
                    "numbering_offset_note": offset_note,
                }
            )
        )
        if acc:
            sources.append(
                Source(
                    kind="UniProtKB",
                    record_id=str(acc),
                    url=f"https://www.uniprot.org/uniprotkb/{acc}/entry",
                    retrieved_at=now,
                )
            )
        if pdb_id:
            sources.append(
                Source(
                    kind="RCSB PDB",
                    record_id=str(pdb_id),
                    url=f"https://www.rcsb.org/structure/{pdb_id}",
                    retrieved_at=now,
                )
            )

    built = BuiltFasta(
        fasta_text=fasta_text,
        chains=enriched,
        is_multichain=len(enriched) > 1,
    )
    return Cited[BuiltFasta](found=True, data=built, sources=sources)


def build_chai_restraints(
    fasta_text: str,
    restraints: list[dict[str, Any]],
) -> Cited[BuiltRestraints]:
    """Build and validate a chai-lab .restraints CSV against a FASTA.

    Accepts either sequential chain letters ('A', 'B') or FASTA entity names
    (automatically resolved to 'A', 'B' with an explicit warning), and refuses
    any residue mismatch, out-of-bounds index, or malformed syntax.
    """
    now = _now_iso()
    chains, fasta_issues = parse_chai_fasta(fasta_text)
    if any(i.severity == "error" for i in fasta_issues):
        return Cited[BuiltRestraints](
            found=False,
            data=None,
            note="FASTA failed validation: " + "; ".join(i.message for i in fasta_issues),
        )
    if not restraints:
        return Cited[BuiltRestraints](
            found=False,
            data=None,
            note="Restraint list is empty",
        )

    chains_by_id = {c.chai_chain_id: c for c in chains}
    chains_by_name = {c.name: c.chai_chain_id for c in chains}
    warnings: list[str] = []

    buf = io.StringIO()
    fieldnames = [
        "restraint_id",
        "chainA",
        "res_idxA",
        "chainB",
        "res_idxB",
        "connection_type",
        "confidence",
        "min_distance_angstrom",
        "max_distance_angstrom",
        "comment",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    built_items: list[RestraintCheck] = []
    for idx, r in enumerate(restraints):
        rid = str(r.get("restraint_id") or f"restraint_{idx}")
        raw_a = str(r.get("chainA", "")).strip()
        raw_b = str(r.get("chainB", "")).strip()

        chain_a = (
            chains_by_name[raw_a]
            if raw_a not in chains_by_id and raw_a in chains_by_name
            else raw_a
        )
        chain_b = (
            chains_by_name[raw_b]
            if raw_b not in chains_by_id and raw_b in chains_by_name
            else raw_b
        )
        if chain_a != raw_a:
            warnings.append(f"{rid}: mapped FASTA entity name '{raw_a}' -> chain '{chain_a}'")
        if chain_b != raw_b:
            warnings.append(f"{rid}: mapped FASTA entity name '{raw_b}' -> chain '{chain_b}'")

        ctype_raw = str(r.get("connection_type", "contact")).strip().lower()
        res_a = str(r.get("res_idxA", "")).strip()
        res_b = str(r.get("res_idxB", "")).strip()
        min_d = float(r.get("min_distance_angstrom", 0.0))
        max_d = float(r.get("max_distance_angstrom", 10.0))
        conf = float(r.get("confidence", 1.0))
        comment = str(r.get("comment", ""))
        translated_from = r.get("translated_from")

        row_dict = {
            "restraint_id": rid,
            "chainA": chain_a,
            "res_idxA": res_a,
            "chainB": chain_b,
            "res_idxB": res_b,
            "connection_type": ctype_raw,
            "confidence": conf,
            "min_distance_angstrom": min_d,
            "max_distance_angstrom": max_d,
            "comment": comment,
        }
        writer.writerow(row_dict)
        if ctype_raw in VALID_CONNECTION_TYPES:
            ctype: RestraintKind = ctype_raw  # type: ignore[assignment]
            built_items.append(
                RestraintCheck(
                    restraint_id=rid,
                    connection_type=ctype,
                    chainA=chain_a,
                    res_idxA=res_a,
                    chainB=chain_b,
                    res_idxB=res_b,
                    min_distance_angstrom=min_d,
                    max_distance_angstrom=max_d,
                    confidence=conf,
                    comment=comment,
                    translated_from=str(translated_from) if translated_from else None,
                )
            )

    csv_text = buf.getvalue()
    val = validate_chai_inputs(fasta_text, csv_text)
    assert val.data is not None
    if not val.data.valid:
        errs = "; ".join(i.message for i in val.data.issues if i.severity == "error")
        return Cited[BuiltRestraints](
            found=False,
            data=None,
            note=f"Restraints failed pre-flight validation: {errs}",
            warnings=warnings,
            sources=val.sources,
        )

    return Cited[BuiltRestraints](
        found=True,
        data=BuiltRestraints(csv_text=csv_text, restraints=built_items),
        warnings=warnings,
        sources=[
            Source(
                kind="chai_lab_spec",
                record_id="chai-lab/data/parsing/restraints.py",
                url="https://github.com/chaidiscovery/chai-lab",
                retrieved_at=now,
            )
        ],
    )
