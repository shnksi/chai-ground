"""Optional 1-command Modal GPU runner for real Chai-1 inference + chai-ground pre/post audit.

Usage (requires `pip install modal` and `modal setup`):
    modal run scripts/modal_fold.py --fasta examples/insulin.fasta --restraints examples/insulin.restraints

Runs pre-flight `validate_chai_inputs` locally BEFORE spinning up a GPU, runs `chai-lab fold`
on an L40S GPU capturing full logs, downloads `scores.model_idx_*.npz` + `.cif` + `chai.log`,
and runs `inspect_chai_outputs` locally so silent failures are never hidden.
"""

from __future__ import annotations

from pathlib import Path

try:
    import modal
except ImportError as exc:  # pragma: no cover - optional cloud dependency
    raise SystemExit("Install Modal first: pip install modal && modal setup") from exc

app = modal.App("chai-ground-fold")

chai_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "kalign")
    .pip_install("chai_lab==0.6.1")
)


@app.function(gpu="L40S", image=chai_image, timeout=1800)
def fold_remote(
    fasta_text: str,
    restraints_csv: str | None = None,
    use_msa_server: bool = False,
) -> dict[str, bytes]:
    """Run chai-lab fold on GPU and return output files + captured stdout/stderr log."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fasta_path = root / "input.fasta"
        fasta_path.write_text(fasta_text)
        out_dir = root / "out"
        out_dir.mkdir()

        cmd = ["chai-lab", "fold", str(fasta_path), str(out_dir)]
        if restraints_csv:
            r_path = root / "input.restraints"
            r_path.write_text(restraints_csv)
            cmd.extend(["--constraint-path", str(r_path)])
        if use_msa_server:
            cmd.append("--use-msa-server")

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        log_text = (
            f"CMD: {' '.join(cmd)}\nRETURN_CODE: {proc.returncode}\n\n"
            f"--- STDOUT ---\n{proc.stdout}\n\n--- STDERR ---\n{proc.stderr}\n"
        )
        artifacts: dict[str, bytes] = {"chai.log": log_text.encode()}
        for p in sorted(out_dir.iterdir()):
            if p.is_file() and (p.suffix in (".npz", ".cif")):
                artifacts[p.name] = p.read_bytes()
        return artifacts


@app.local_entrypoint()
def main(
    fasta: str,
    restraints: str | None = None,
    out_dir: str = "chai_out",
    use_msa_server: bool = False,
) -> None:
    from chai_ground.inputs import validate_chai_inputs
    from chai_ground.outputs import inspect_chai_outputs

    fasta_text = Path(fasta).read_text()
    restraints_csv = Path(restraints).read_text() if restraints else None

    # 1. Pre-flight check BEFORE paying for GPU time
    check = validate_chai_inputs(fasta_text, restraints_csv)
    assert check.data is not None
    if not check.data.valid:
        print("ABORTED before GPU launch — pre-flight validation failed:")
        for issue in check.data.issues:
            print(f"  [{issue.code}] {issue.location}: {issue.message}")
        raise SystemExit(1)

    print(
        f"Pre-flight passed ({len(check.data.chains)} chains, {check.data.restraint_count} restraints). Launching L40S..."
    )
    artifacts = fold_remote.remote(fasta_text, restraints_csv, use_msa_server=use_msa_server)

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (dest / name).write_bytes(content)

    # 2. Post-run honesty audit
    audit = inspect_chai_outputs(dest)
    print(audit.model_dump_json(indent=2))
