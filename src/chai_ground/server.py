"""MCP stdio server exposing chai-ground's four deterministic Chai-1 grounding tools."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from chai_ground import __version__
from chai_ground.models import BuiltFasta, BuiltRestraints, Cited, OutputAudit, ValidationReport
from chai_ground.tools import (
    tool_build_chai_fasta,
    tool_build_chai_restraints,
    tool_inspect_chai_outputs,
    tool_validate_chai_inputs,
)

INSTRUCTIONS = (
    "Pre-flight validator and output-honesty auditor for Chai-1 (chai-lab). "
    "Every tool returns a Cited object with found/valid status and explicit notes. "
    "Never guess chain IDs, residue indices, or Chai-1 confidence scores when a check fails."
)


def build_server() -> MCPServer:
    mcp = MCPServer("chai-ground", instructions=INSTRUCTIONS, version=__version__)
    read_only = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @mcp.tool(annotations=read_only)
    def build_chai_fasta(entities: list[dict[str, Any]]) -> Cited[BuiltFasta]:
        """Build and validate a Chai-1 FASTA from structured entity dicts.

        Each entity dict requires: entity_type ('protein'|'ligand'|'dna'|'rna'|'glycan'),
        name (str), sequence (str). Returns sequential chai-lab chain letters (A, B, C...)
        or found=false with the exact validation failure.
        """
        return tool_build_chai_fasta(entities)

    @mcp.tool(annotations=read_only)
    def build_chai_restraints(
        fasta_text: str,
        restraints: list[dict[str, Any]],
    ) -> Cited[BuiltRestraints]:
        """Build and pre-flight validate a Chai-1 .restraints CSV against a FASTA.

        Maps FASTA entity names to sequential chain IDs (A, B, ...) with an explicit warning,
        verifies 1-based residue positions and exact wild-type residue letters, and refuses
        specs that chai-lab would silently drop.
        """
        return tool_build_chai_restraints(fasta_text, restraints)

    @mcp.tool(annotations=read_only)
    def validate_chai_inputs(
        fasta_text: str,
        restraints_csv: str | None = None,
    ) -> Cited[ValidationReport]:
        """Validate a Chai-1 FASTA and optional restraints CSV before GPU inference.

        Catches sequential chain-ID bugs (#413/#434), residue-letter mismatches and
        author-vs-FASTA numbering drift (#435), mismatched modified-residue brackets (#439),
        invalid SMILES, and malformed pocket/contact/covalent rules.
        """
        return tool_validate_chai_inputs(fasta_text, restraints_csv)

    @mcp.tool(annotations=read_only)
    def inspect_chai_outputs(
        output_dir: str,
        log_path: str | None = None,
    ) -> Cited[OutputAudit]:
        """Audit a Chai-1 output directory (scores.model_idx_*.npz + optional chai.log).

        Ranks all diffusion samples properly, flags single-chain ipTM caveats (#431),
        detects inter-chain steric clashes, and scans run logs for silently dropped restraints
        or failed template loads.
        """
        return tool_inspect_chai_outputs(output_dir, log_path)

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
