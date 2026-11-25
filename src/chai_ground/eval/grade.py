"""Deterministic graders (no LLM-as-a-judge) + gold verification."""

from __future__ import annotations

from pathlib import Path

from chai_ground.eval.schema import AgentSubmission, CaseGrade, EvalCase, EvalSummary
from chai_ground.tools import tool_inspect_chai_outputs, tool_validate_chai_inputs


def grade_case(
    case: EvalCase,
    submission: AgentSubmission,
    *,
    fetched_records: set[str],
    tool_calls: int = 0,
    cost_usd: float = 0.0,
) -> CaseGrade:
    """Deterministically grade a single eval submission."""
    verdict_correct = submission.verdict == case.expected_verdict
    reasons: list[str] = []

    if not verdict_correct:
        reasons.append(f"verdict={submission.verdict}, expected={case.expected_verdict}")

    details_correct = True
    if case.expected_issue_codes:
        submitted_codes = set(submission.issue_codes)
        missing_codes = [c for c in case.expected_issue_codes if c not in submitted_codes]
        if missing_codes:
            details_correct = False
            reasons.append(f"missing expected issue code(s): {missing_codes}")

    if (
        case.expected_best_model_idx is not None
        and submission.best_model_idx != case.expected_best_model_idx
    ):
        details_correct = False
        reasons.append(
            f"best_model_idx={submission.best_model_idx}, expected={case.expected_best_model_idx}"
        )

    if case.expected_metric_name is not None:
        if submission.reported_metric_name != case.expected_metric_name:
            details_correct = False
            reasons.append(
                f"metric_name={submission.reported_metric_name}, expected={case.expected_metric_name}"
            )
        if case.expected_metric_value is not None:
            if submission.reported_metric_value is None:
                details_correct = False
                reasons.append(f"missing metric value (expected {case.expected_metric_value})")
            elif abs(submission.reported_metric_value - case.expected_metric_value) > 0.005:
                details_correct = False
                reasons.append(
                    f"metric_value={submission.reported_metric_value}, expected={case.expected_metric_value}"
                )

    # Grounded means at least one cited record was actually returned by a tool call in this episode.
    grounded = bool(submission.cited_records) and any(
        rec in fetched_records for rec in submission.cited_records
    )

    passed = verdict_correct and details_correct
    return CaseGrade(
        case_id=case.case_id,
        family=case.family,
        is_trap=case.is_trap,
        passed=passed,
        verdict_correct=verdict_correct,
        details_correct=details_correct,
        grounded=grounded,
        failure_reason="; ".join(reasons) if reasons else None,
        tool_calls=tool_calls,
        cost_usd=cost_usd,
        submission=submission,
    )


def solve_case_with_tools(
    case: EvalCase,
    fixture_paths: dict[str, Path],
) -> tuple[AgentSubmission, set[str], int]:
    """Deterministic tool-using oracle (also verifies 100% gold reachability via tools)."""
    fetched: set[str] = set()
    calls = 0

    if case.output_fixture_name is not None:
        out_dir = fixture_paths[case.output_fixture_name]
        res = tool_inspect_chai_outputs(str(out_dir))
        calls += 1
        for src in res.sources:
            fetched.add(src.record_id)
        assert res.found and res.data is not None
        audit = res.data
        best = audit.samples_ranked[0]
        metric_name = "ptm" if audit.is_single_chain else "iptm"
        metric_val = (
            best.ptm
            if audit.is_single_chain
            else (best.iptm if best.iptm is not None else best.raw_iptm)
        )
        submission = AgentSubmission(
            verdict="accept" if audit.trustworthy else "reject",
            best_model_idx=audit.best_model_idx,
            reported_metric_name=metric_name,
            reported_metric_value=metric_val,
            explanation="; ".join(audit.verdict_reasons),
            cited_records=[s.record_id for s in res.sources],
        )
        return submission, fetched, calls

    assert case.fasta_text is not None
    res_val = tool_validate_chai_inputs(case.fasta_text, case.restraints_csv)
    calls += 1
    for src in res_val.sources:
        fetched.add(src.record_id)
    assert res_val.found and res_val.data is not None
    report = res_val.data
    codes = [i.code for i in report.issues if i.severity == "error"]
    submission = AgentSubmission(
        verdict="accept" if report.valid else "reject",
        issue_codes=codes,
        explanation="Valid Chai-1 inputs"
        if report.valid
        else "; ".join(i.message for i in report.issues),
        cited_records=[s.record_id for s in res_val.sources],
    )
    return submission, fetched, calls


def summarize_grades(arm: str, model: str, grades: list[CaseGrade]) -> EvalSummary:
    valid = [g for g in grades if not g.is_trap]
    traps = [g for g in grades if g.is_trap]
    valid_acc = sum(g.passed for g in valid) / len(valid) if valid else 0.0
    trap_caught = sum(g.verdict_correct for g in traps) / len(traps) if traps else 0.0
    false_alarms = (
        sum(g.submission.verdict == "reject" for g in valid) / len(valid) if valid else 0.0
    )
    overall = sum(g.passed for g in grades) / len(grades) if grades else 0.0
    grounded = sum(g.grounded for g in grades) / len(grades) if grades else 0.0

    by_family: dict[str, dict[str, object]] = {}
    for fam in sorted({g.family for g in grades}):
        fg = [g for g in grades if g.family == fam]
        by_family[fam] = {
            "total": len(fg),
            "passed": sum(g.passed for g in fg),
            "accuracy": round(sum(g.passed for g in fg) / len(fg), 4),
        }

    return EvalSummary(
        arm=arm,
        model=model,
        total_cases=len(grades),
        valid_cases=len(valid),
        trap_cases=len(traps),
        valid_accuracy=round(valid_acc, 4),
        trap_caught_rate=round(trap_caught, 4),
        false_alarm_rate=round(false_alarms, 4),
        overall_accuracy=round(overall, 4),
        grounded_rate=round(grounded, 4),
        total_tool_calls=sum(g.tool_calls for g in grades),
        total_cost_usd=round(sum(g.cost_usd for g in grades), 6),
        by_family=by_family,
    )
