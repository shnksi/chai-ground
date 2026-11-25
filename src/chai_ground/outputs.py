"""Output-honesty auditor for Chai-1 (chai-lab) output folders.

Reads scores.model_idx_*.npz (exact keys from chai_lab.ranking.rank.get_scores),
pred.model_idx_*.cif presence, and optional chai.log stderr/stdout to catch:
1. Reporting model_idx_0 when a different diffusion sample ranked higher.
2. Citing ipTM on a single-chain monomer (where ipTM is undefined/0 and aggregate_score
   = 0.2*pTM + 0.8*0 understates monomer confidence; chai-lab PR #431).
3. Ignoring inter-chain steric clashes (has_inter_chain_clashes=True).
4. Reporting a fold as restraint-guided when chai-lab's log silently dropped restraints (#413/#435).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from chai_ground.models import Cited, ModelSampleScore, OutputAudit, Source

SCORE_FILE_RE = re.compile(r"^scores\.model_idx_(\d+)\.npz$")
REQUIRED_NPZ_KEYS = (
    "aggregate_score",
    "ptm",
    "iptm",
    "per_chain_ptm",
    "has_inter_chain_clashes",
    "chain_chain_clashes",
)

# Exact silent-drop log patterns from chai-lab source
RESTRAINT_DROP_PATTERNS = (
    re.compile(r"Error .* generating distance constraints", re.IGNORECASE),
    re.compile(r"Cannot place restraint", re.IGNORECASE),
    re.compile(r"dropping restraint", re.IGNORECASE),
    re.compile(r"Error .* generating pocket constraints", re.IGNORECASE),
)
TEMPLATE_ERROR_PATTERNS = (
    re.compile(r"Failed to load entity data for TemplateHit", re.IGNORECASE),
    re.compile(r"Skipping .* as it failed to tokenize", re.IGNORECASE),
)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scalar_float(arr: np.ndarray) -> float:
    return float(np.asarray(arr).reshape(-1)[0])


def _scalar_bool(arr: np.ndarray) -> bool:
    return bool(np.asarray(arr).reshape(-1)[0])


def inspect_chai_outputs(
    output_dir: str | Path,
    log_path: str | Path | None = None,
) -> Cited[OutputAudit]:
    """Audit a chai-lab output directory and return grounded rankings + honesty flags."""
    now = _now_iso()
    root = Path(output_dir)
    if not root.exists() or not root.is_dir():
        return Cited[OutputAudit](
            found=False,
            data=None,
            note=f"Output directory does not exist: {root}",
        )

    score_files: list[tuple[int, Path]] = []
    for p in sorted(root.iterdir()):
        m = SCORE_FILE_RE.match(p.name)
        if m:
            score_files.append((int(m.group(1)), p))

    if not score_files:
        return Cited[OutputAudit](
            found=False,
            data=None,
            note=f"No scores.model_idx_*.npz files found in {root}",
        )

    samples: list[ModelSampleScore] = []
    sources: list[Source] = []

    for model_idx, npz_path in sorted(score_files, key=lambda x: x[0]):
        try:
            with np.load(npz_path) as data:
                missing = [k for k in REQUIRED_NPZ_KEYS if k not in data]
                if missing:
                    return Cited[OutputAudit](
                        found=False,
                        data=None,
                        note=f"{npz_path.name} is missing required Chai-1 ranking keys: {missing}",
                    )
                agg = _scalar_float(data["aggregate_score"])
                ptm = _scalar_float(data["ptm"])
                raw_iptm = _scalar_float(data["iptm"])
                per_chain = [float(x) for x in np.asarray(data["per_chain_ptm"]).reshape(-1)]
                has_clash = _scalar_bool(data["has_inter_chain_clashes"])
                clash_mat = np.asarray(data["chain_chain_clashes"])
        except Exception as exc:
            return Cited[OutputAudit](
                found=False,
                data=None,
                note=f"Failed to read {npz_path.name}: {exc}",
            )

        is_single = len(per_chain) == 1
        clashing_pairs: list[tuple[int, int]] = []
        if clash_mat.ndim >= 2:
            mat2d = clash_mat[0] if clash_mat.ndim == 3 else clash_mat
            rows, cols = mat2d.shape[:2]
            for i in range(rows):
                for j in range(i + 1, cols):
                    if float(mat2d[i, j]) > 0:
                        clashing_pairs.append((i, j))

        cif_present = (root / f"pred.model_idx_{model_idx}.cif").is_file()
        samples.append(
            ModelSampleScore(
                model_idx=model_idx,
                aggregate_score=round(agg, 6),
                ptm=round(ptm, 6),
                iptm=None if is_single else round(raw_iptm, 6),
                raw_iptm=round(raw_iptm, 6),
                per_chain_ptm=[round(x, 6) for x in per_chain],
                has_inter_chain_clashes=has_clash,
                clashing_chain_pairs=clashing_pairs,
                cif_present=cif_present,
                is_single_chain=is_single,
            )
        )
        sources.append(
            Source(
                kind="local_file",
                record_id=str(npz_path),
                retrieved_at=now,
            )
        )

    is_single_chain = all(s.is_single_chain for s in samples)
    # Sort descending by ptm for single-chain monomers (since aggregate_score = 0.2*pTM + 0.8*0),
    # and by aggregate_score for multi-chain complexes (which already penalizes clashes by -100).
    if is_single_chain:
        ranked = sorted(samples, key=lambda s: (s.ptm, -s.model_idx), reverse=True)
    else:
        ranked = sorted(samples, key=lambda s: (s.aggregate_score, -s.model_idx), reverse=True)

    best = ranked[0]
    model_0_is_best = best.model_idx == 0

    # Check log for silent restraint or template drops
    resolved_log: Path | None = None
    if log_path is not None:
        p = Path(log_path)
        if p.is_file():
            resolved_log = p
    else:
        for candidate in ("chai.log", "run.log", "stderr.log"):
            p = root / candidate
            if p.is_file():
                resolved_log = p
                break

    restraints_dropped = False
    template_errors = False
    log_issues: list[str] = []
    if resolved_log is not None:
        sources.append(Source(kind="local_file", record_id=str(resolved_log), retrieved_at=now))
        for line in resolved_log.read_text(errors="replace").splitlines():
            if any(pat.search(line) for pat in RESTRAINT_DROP_PATTERNS):
                restraints_dropped = True
                log_issues.append(line.strip()[:240])
            elif any(pat.search(line) for pat in TEMPLATE_ERROR_PATTERNS):
                template_errors = True
                log_issues.append(line.strip()[:240])

    single_chain_caveat: str | None = None
    if is_single_chain:
        single_chain_caveat = (
            "Single-chain monomer: ipTM is not applicable (Chai-1 sets aggregate_score = 0.2*pTM + 0.8*0). "
            "Report pTM rather than ipTM or raw aggregate_score."
        )

    verdict_reasons: list[str] = []
    trustworthy = True
    if best.has_inter_chain_clashes:
        trustworthy = False
        verdict_reasons.append(
            f"Best sample (model_idx_{best.model_idx}) has inter-chain steric clashes "
            f"(pairs={best.clashing_chain_pairs})"
        )
    if restraints_dropped:
        trustworthy = False
        verdict_reasons.append(
            "Run log indicates restraints were dropped during featurization; prediction may be unconstrained"
        )
    if template_errors:
        verdict_reasons.append(
            "Run log recorded template loading failures; verify whether intended templates were applied"
        )
    if not best.cif_present:
        verdict_reasons.append(
            f"pred.model_idx_{best.model_idx}.cif not found next to scores.model_idx_{best.model_idx}.npz"
        )
    if is_single_chain:
        if best.ptm < 0.5:
            trustworthy = False
            verdict_reasons.append(f"Low monomer confidence: best pTM={best.ptm:.3f} < 0.50")
    else:
        if best.iptm is not None and best.iptm < 0.5:
            verdict_reasons.append(
                f"Low interface confidence: best ipTM={best.iptm:.3f} < 0.50 (pTM={best.ptm:.3f})"
            )

    if trustworthy and not verdict_reasons:
        verdict_reasons.append(
            f"Best sample is model_idx_{best.model_idx} with no inter-chain clashes and clean logs"
        )

    audit = OutputAudit(
        output_dir=str(root),
        num_samples=len(samples),
        is_single_chain=is_single_chain,
        samples_ranked=ranked,
        best_model_idx=best.model_idx,
        model_idx_0_is_best=model_0_is_best,
        any_best_sample_clashes=best.has_inter_chain_clashes,
        restraints_dropped_in_log=restraints_dropped,
        template_load_errors_in_log=template_errors,
        log_issues=log_issues,
        single_chain_iptm_caveat=single_chain_caveat,
        trustworthy=trustworthy,
        verdict_reasons=verdict_reasons,
    )
    return Cited[OutputAudit](found=True, data=audit, sources=sources)
