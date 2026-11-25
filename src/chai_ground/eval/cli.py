"""Eval CLI: check-gold, offline stub + decoy runs, and optional budgeted live Anthropic run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from chai_ground.eval.fixtures import build_eval_cases, materialize_output_fixtures
from chai_ground.eval.grade import grade_case, solve_case_with_tools, summarize_grades
from chai_ground.eval.schema import AgentSubmission, CaseGrade, EvalCase

ArmKind = Literal["tools_stub", "decoy_always_accept", "decoy_always_reject"]


def run_offline_arm(
    arm: ArmKind,
    cases: list[EvalCase],
    fixture_paths: dict[str, Path],
) -> list[CaseGrade]:
    grades: list[CaseGrade] = []
    for case in cases:
        if arm == "tools_stub":
            sub, fetched, calls = solve_case_with_tools(case, fixture_paths)
        elif arm == "decoy_always_accept":
            sub = AgentSubmission(verdict="accept", explanation="decoy: always accept")
            fetched = set()
            calls = 0
        elif arm == "decoy_always_reject":
            sub = AgentSubmission(verdict="reject", explanation="decoy: always reject")
            fetched = set()
            calls = 0
        grades.append(
            grade_case(case, sub, fetched_records=fetched, tool_calls=calls, cost_usd=0.0)
        )
    return grades


def cmd_check_gold(fixtures_dir: Path) -> int:
    fixture_paths = materialize_output_fixtures(fixtures_dir)
    cases = build_eval_cases()
    grades = run_offline_arm("tools_stub", cases, fixture_paths)
    failed = [g for g in grades if not g.passed or not g.grounded]
    print(f"Gold check: {len(grades) - len(failed)}/{len(grades)} passed & grounded")
    for g in failed:
        print(f"  FAIL {g.case_id}: {g.failure_reason} (grounded={g.grounded})")
    return 0 if not failed else 1


def cmd_dry_run(fixtures_dir: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = materialize_output_fixtures(fixtures_dir)
    cases = build_eval_cases()

    # Export canonical questions.jsonl
    (out_dir / "cases.jsonl").write_text("\n".join(c.model_dump_json() for c in cases) + "\n")

    summaries = []
    for arm in ("tools_stub", "decoy_always_accept", "decoy_always_reject"):
        grades = run_offline_arm(arm, cases, fixture_paths)
        summary = summarize_grades(arm=arm, model="offline-deterministic", grades=grades)
        summaries.append(summary)
        (out_dir / f"{arm}-episodes.jsonl").write_text(
            "\n".join(g.model_dump_json() for g in grades) + "\n"
        )
        (out_dir / f"{arm}-summary.json").write_text(summary.model_dump_json(indent=2) + "\n")

    n_valid = summaries[0].valid_cases
    n_traps = summaries[0].trap_cases
    lines = [
        "# chai-ground Offline Dry-Run & Decoy Bracket Report",
        "",
        f"| Arm | Valid correct ({n_valid}) | Traps caught ({n_traps}) "
        "| False alarms | Overall | Grounded | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| `{s.arm}` | {s.valid_accuracy * 100:.1f}% | {s.trap_caught_rate * 100:.1f}% | "
            f"{s.false_alarm_rate * 100:.1f}% | {s.overall_accuracy * 100:.1f}% | "
            f"{s.grounded_rate * 100:.1f}% | ${s.total_cost_usd:.2f} |"
        )
    report_path = out_dir / "DRY_RUN_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(report_path.read_text())
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chai-ground-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gold = sub.add_parser("check-gold", help="Verify all gold cases are reachable via tools")
    p_gold.add_argument("--fixtures-dir", type=Path, default=Path("evals/fixtures"))

    p_dry = sub.add_parser("dry-run", help="Run zero-cost stub + decoy bracket verification")
    p_dry.add_argument("--fixtures-dir", type=Path, default=Path("evals/fixtures"))
    p_dry.add_argument("--out", type=Path, default=Path("evals/dry-run"))

    args = parser.parse_args(argv)
    if args.cmd == "check-gold":
        sys.exit(cmd_check_gold(args.fixtures_dir))
    elif args.cmd == "dry-run":
        sys.exit(cmd_dry_run(args.fixtures_dir, args.out))


if __name__ == "__main__":
    main()
