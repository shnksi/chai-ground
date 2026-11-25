"""Typed models for chai-ground. Every tool returns Cited[T] with explicit found/valid status."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field

SourceKind = Literal["UniProtKB", "RCSB PDB", "local_file", "chai_lab_spec"]
EntityKind = Literal["protein", "ligand", "dna", "rna", "glycan"]
RestraintKind = Literal["contact", "pocket", "covalent"]
Severity = Literal["error", "warning"]
T = TypeVar("T")


class Source(BaseModel):
    kind: SourceKind
    record_id: str = Field(description="Accession, PDB ID, file path, or spec rule")
    url: str | None = Field(default=None, description="Reference URL when applicable")
    retrieved_at: str


class Cited(BaseModel, Generic[T]):
    """Uniform tool result wrapper: missing/invalid inputs return found=False with note."""

    found: bool
    data: T | None = None
    note: str | None = None
    warnings: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)


class ChainMapping(BaseModel):
    """Maps a biological entity to the sequential chai-lab chain ID (A, B, C, ...)."""

    chai_chain_id: str = Field(description="Sequential chai-lab chain letter: A, B, C, ...")
    entity_type: EntityKind
    name: str = Field(description="FASTA entity name (after 'protein|name=')")
    sequence: str = Field(description="Validated FASTA sequence or SMILES string")
    length: int = Field(description="Token/residue count for polymers, character length for SMILES")
    uniprot_accession: str | None = None
    pdb_id: str | None = None
    pdb_chain: str | None = None
    numbering_offset_note: str | None = Field(
        default=None,
        description="Explicit warning when PDB author numbering differs from 1-based FASTA index",
    )


class BuiltFasta(BaseModel):
    fasta_text: str
    chains: list[ChainMapping]
    is_multichain: bool


class RestraintCheck(BaseModel):
    restraint_id: str
    connection_type: RestraintKind
    chainA: str
    res_idxA: str
    chainB: str
    res_idxB: str
    min_distance_angstrom: float
    max_distance_angstrom: float
    confidence: float
    comment: str = ""
    translated_from: str | None = Field(
        default=None,
        description="Original biological numbering if translated from PDB author numbering",
    )


class BuiltRestraints(BaseModel):
    csv_text: str
    restraints: list[RestraintCheck]


class ValidationIssue(BaseModel):
    severity: Severity
    code: str = Field(description="Machine-readable failure code, e.g. CHAIN_ID_NOT_SEQUENTIAL")
    message: str
    location: str | None = None


class ValidationReport(BaseModel):
    valid: bool
    chains: list[ChainMapping] = Field(default_factory=list)
    restraint_count: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)


class ModelSampleScore(BaseModel):
    """Exact fields emitted by chai_lab.ranking.rank.get_scores into scores.model_idx_*.npz."""

    model_idx: int
    aggregate_score: float
    ptm: float
    iptm: float | None = Field(
        description="Null on single-chain monomers where ipTM is not an interface metric"
    )
    raw_iptm: float
    per_chain_ptm: list[float]
    has_inter_chain_clashes: bool
    clashing_chain_pairs: list[tuple[int, int]] = Field(default_factory=list)
    cif_present: bool = False
    is_single_chain: bool = False


class OutputAudit(BaseModel):
    output_dir: str
    num_samples: int
    is_single_chain: bool
    samples_ranked: list[ModelSampleScore]
    best_model_idx: int
    model_idx_0_is_best: bool
    any_best_sample_clashes: bool
    restraints_dropped_in_log: bool
    template_load_errors_in_log: bool
    log_issues: list[str] = Field(default_factory=list)
    single_chain_iptm_caveat: str | None = None
    trustworthy: bool
    verdict_reasons: list[str] = Field(default_factory=list)
