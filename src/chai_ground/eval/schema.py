"""Pydantic schemas for the preregistered chai-ground agent failure-mode eval."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CaseFamily = Literal[
    "chain_id_trap",
    "numbering_offset_trap",
    "fasta_syntax_trap",
    "output_ranking_honesty",
    "single_chain_iptm_trap",
    "clash_and_log_audit",
]

Verdict = Literal["accept", "reject"]


class EvalCase(BaseModel):
    case_id: str
    family: CaseFamily
    is_trap: bool = Field(
        description="True when the input or Chai-1 run has a silent defect and MUST be rejected"
    )
    prompt: str
    fasta_text: str | None = None
    restraints_csv: str | None = None
    output_fixture_name: str | None = None
    expected_verdict: Verdict
    expected_issue_codes: list[str] = Field(default_factory=list)
    expected_best_model_idx: int | None = None
    expected_metric_name: Literal["ptm", "iptm", "aggregate_score"] | None = None
    expected_metric_value: float | None = None
    notes: str = ""


class AgentSubmission(BaseModel):
    verdict: Verdict
    issue_codes: list[str] = Field(default_factory=list)
    best_model_idx: int | None = None
    reported_metric_name: str | None = None
    reported_metric_value: float | None = None
    explanation: str = ""
    cited_records: list[str] = Field(default_factory=list)


class CaseGrade(BaseModel):
    case_id: str
    family: CaseFamily
    is_trap: bool
    passed: bool
    verdict_correct: bool
    details_correct: bool
    grounded: bool
    failure_reason: str | None = None
    tool_calls: int = 0
    cost_usd: float = 0.0
    submission: AgentSubmission


class EvalSummary(BaseModel):
    arm: str
    model: str
    total_cases: int
    valid_cases: int
    trap_cases: int
    valid_accuracy: float
    trap_caught_rate: float
    false_alarm_rate: float
    overall_accuracy: float
    grounded_rate: float
    total_tool_calls: int
    total_cost_usd: float
    by_family: dict[str, dict[str, Any]]
