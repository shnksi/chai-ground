"""Single tool surface shared between the MCP server and the eval harness.

Every tool returns a Cited[...] JSON-serializable object so what gets measured is what ships.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chai_ground.inputs import (
    build_chai_fasta_from_entities,
    build_chai_restraints,
    validate_chai_inputs,
)
from chai_ground.models import (
    BuiltFasta,
    BuiltRestraints,
    Cited,
    OutputAudit,
    ValidationReport,
)
from chai_ground.outputs import inspect_chai_outputs


def tool_build_chai_fasta(entities: list[dict[str, Any]]) -> Cited[BuiltFasta]:
    """Build and validate a Chai-1 FASTA file with explicit sequential chain IDs (A, B, C...)."""
    return build_chai_fasta_from_entities(entities)


def tool_build_chai_restraints(
    fasta_text: str,
    restraints: list[dict[str, Any]],
) -> Cited[BuiltRestraints]:
    """Build and pre-flight validate a Chai-1 .restraints CSV against a FASTA."""
    return build_chai_restraints(fasta_text, restraints)


def tool_validate_chai_inputs(
    fasta_text: str,
    restraints_csv: str | None = None,
) -> Cited[ValidationReport]:
    """Pre-flight check a Chai-1 FASTA and optional restraints CSV for silent failure modes."""
    return validate_chai_inputs(fasta_text, restraints_csv)


def tool_inspect_chai_outputs(
    output_dir: str,
    log_path: str | None = None,
) -> Cited[OutputAudit]:
    """Audit a Chai-1 output directory (scores.model_idx_*.npz + logs) with honesty flags."""
    return inspect_chai_outputs(Path(output_dir), Path(log_path) if log_path else None)
