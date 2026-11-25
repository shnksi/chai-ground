"""Offline test suite for chai-ground: pre-flight validation, output auditing, MCP tools & eval."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chai_ground.eval.cli import cmd_check_gold, cmd_dry_run, run_offline_arm
from chai_ground.eval.fixtures import (
    HBB_UNIPROT_25,
    IMATINIB_SMILES,
    INSULIN_A_21,
    INSULIN_B_30,
    KRAS_G12C_40,
    KRAS_WT_40,
    WARHEAD_SMILES,
    build_eval_cases,
    materialize_output_fixtures,
)
from chai_ground.eval.grade import summarize_grades
from chai_ground.inputs import (
    build_chai_fasta_from_entities,
    build_chai_restraints,
    parse_modified_fasta_tokens,
    validate_chai_inputs,
    validate_smiles,
)
from chai_ground.outputs import inspect_chai_outputs
from chai_ground.server import build_server
from chai_ground.tools import (
    tool_build_chai_fasta,
    tool_build_chai_restraints,
    tool_inspect_chai_outputs,
    tool_validate_chai_inputs,
)


def test_modified_fasta_rejects_mismatched_brackets() -> None:
    tokens, err = parse_modified_fasta_tokens("(SEP)[BAL]ACDE")
    assert err is None
    assert tokens == ["SEP", "BAL", "A", "C", "D", "E"]

    tokens_bad, err_bad = parse_modified_fasta_tokens("ACDE(SEP]FG")
    assert tokens_bad is None
    assert err_bad is not None and "Mismatched" in err_bad


def test_smiles_validation() -> None:
    assert validate_smiles(IMATINIB_SMILES) is None
    assert validate_smiles("Cc1ccc(NC(=O)") is not None
    assert validate_smiles("   ") is not None


def test_build_fasta_assigns_sequential_chains() -> None:
    res = build_chai_fasta_from_entities(
        [
            {
                "entity_type": "protein",
                "name": "kras_wt",
                "sequence": KRAS_WT_40,
                "uniprot_accession": "P01116",
            },
            {"entity_type": "ligand", "name": "imatinib", "sequence": IMATINIB_SMILES},
        ]
    )
    assert res.found and res.data is not None
    assert [c.chai_chain_id for c in res.data.chains] == ["A", "B"]
    assert res.data.is_multichain is True
    assert any(s.kind == "UniProtKB" and s.record_id == "P01116" for s in res.sources)


def test_restraint_builder_maps_entity_names_and_catches_author_offset() -> None:
    fasta = (
        f">protein|name=hbb_uniprot\n{HBB_UNIPROT_25}\n>protein|name=insulin_A\n{INSULIN_A_21}\n"
    )

    # Mapping entity names -> A/B succeeds with warning when residue letter matches (E7 in UniProt P68871)
    ok = build_chai_restraints(
        fasta,
        [
            {
                "chainA": "hbb_uniprot",
                "res_idxA": "E7",
                "chainB": "insulin_A",
                "res_idxB": "C6",
                "connection_type": "contact",
                "max_distance_angstrom": 8.0,
            }
        ],
    )
    assert ok.found and ok.data is not None
    assert ok.data.restraints[0].chainA == "A"
    assert ok.data.restraints[0].chainB == "B"
    assert len(ok.warnings) == 2

    # Author numbering E6 on full UniProt P68871 fails loudly because pos 6 is 'P', not 'E'
    bad = build_chai_restraints(
        fasta,
        [
            {
                "chainA": "A",
                "res_idxA": "E6",
                "chainB": "B",
                "res_idxB": "C6",
                "connection_type": "contact",
            }
        ],
    )
    assert not bad.found
    assert bad.note is not None and "not 'E'" in bad.note


def test_validate_catches_issue_413_silent_chain_name_drop() -> None:
    fasta = f">protein|name=sequence_1\n{INSULIN_B_30}\n>protein|name=sequence_2\n{INSULIN_A_21}\n"
    csv_text = (
        "restraint_id,chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
        "min_distance_angstrom,max_distance_angstrom,comment\n"
        "r0,sequence_1,C7,sequence_2,C7,contact,1.0,1.5,3.0,disulfide\n"
    )
    res = validate_chai_inputs(fasta, csv_text)
    assert res.found and res.data is not None
    assert not res.data.valid
    codes = {i.code for i in res.data.issues}
    assert "CHAIN_ID_NOT_SEQUENTIAL" in codes


def test_covalent_restraint_requires_atoms_and_matching_residue() -> None:
    fasta = f">protein|name=kras_g12c\n{KRAS_G12C_40}\n>ligand|name=warhead\n{WARHEAD_SMILES}\n"
    ok_csv = (
        "restraint_id,chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
        "min_distance_angstrom,max_distance_angstrom,comment\n"
        "r0,A,C12@SG,B,@C1,covalent,1.0,0.0,2.5,covalent\n"
    )
    res = validate_chai_inputs(fasta, ok_csv)
    assert res.data is not None and res.data.valid


def test_output_audit_catches_ranking_clashes_single_chain_and_dropped_restraints(
    tmp_path: Path,
) -> None:
    fixtures = materialize_output_fixtures(tmp_path)

    # 1. Best model is idx 3, NOT idx 0
    r1 = inspect_chai_outputs(fixtures["clean_multichain_best_idx3"])
    assert r1.found and r1.data is not None
    assert r1.data.best_model_idx == 3
    assert r1.data.model_idx_0_is_best is False
    assert r1.data.trustworthy is True

    # 2. Single-chain monomer: iptm must be None, caveat populated, ranked by ptm
    r3 = inspect_chai_outputs(fixtures["single_chain_monomer_high_ptm"])
    assert r3.found and r3.data is not None
    assert r3.data.is_single_chain is True
    assert r3.data.best_model_idx == 1
    assert r3.data.samples_ranked[0].iptm is None
    assert r3.data.samples_ranked[0].ptm == 0.92
    assert r3.data.single_chain_iptm_caveat is not None
    assert r3.data.trustworthy is True

    # 3. All clashing -> untrustworthy even with high ptm/iptm
    r5 = inspect_chai_outputs(fixtures["all_samples_clashing"])
    assert r5.found and r5.data is not None
    assert r5.data.trustworthy is False
    assert r5.data.any_best_sample_clashes is True
    assert r5.data.samples_ranked[0].clashing_chain_pairs == [(0, 1)]

    # 4. model_idx_0 clashes -> ranks idx 2 first cleanly
    r6 = inspect_chai_outputs(fixtures["model0_clashes_idx2_clean"])
    assert r6.found and r6.data is not None
    assert r6.data.best_model_idx == 2
    assert r6.data.trustworthy is True

    # 5. Silent restraint drop in log -> untrustworthy despite 0.88 ipTM
    r7 = inspect_chai_outputs(fixtures["silent_restraint_drop_log"])
    assert r7.found and r7.data is not None
    assert r7.data.restraints_dropped_in_log is True
    assert r7.data.trustworthy is False

    # 6. Missing / corrupted dir -> found=False, never fabricates 0.0
    missing = inspect_chai_outputs(tmp_path / "does_not_exist")
    assert not missing.found and missing.data is None

    bad_dir = tmp_path / "corrupt"
    bad_dir.mkdir()
    np.savez(bad_dir / "scores.model_idx_0.npz", ptm=np.array([0.5]))
    corrupt = inspect_chai_outputs(bad_dir)
    assert not corrupt.found and corrupt.note is not None and "missing required" in corrupt.note


def test_mcp_server_and_tool_wrappers_round_trip(tmp_path: Path) -> None:
    server = build_server()
    assert server.name == "chai-ground"

    fixtures = materialize_output_fixtures(tmp_path)
    built = tool_build_chai_fasta(
        [{"entity_type": "protein", "name": "kras", "sequence": KRAS_WT_40}]
    )
    assert built.found and built.data is not None
    val = tool_validate_chai_inputs(built.data.fasta_text)
    assert val.data is not None and val.data.valid is True
    restraints = tool_build_chai_restraints(
        f">protein|name=insB\n{INSULIN_B_30}\n>protein|name=insA\n{INSULIN_A_21}\n",
        [
            {
                "chainA": "A",
                "res_idxA": "C7",
                "chainB": "B",
                "res_idxB": "C7",
                "connection_type": "contact",
            }
        ],
    )
    assert restraints.found is True
    audited = tool_inspect_chai_outputs(str(fixtures["clean_multichain_best_idx3"]))
    assert audited.data is not None and audited.data.best_model_idx == 3


def test_eval_gold_and_decoy_brackets(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    out_dir = tmp_path / "dry-run"
    assert cmd_check_gold(fixtures_dir) == 0
    assert cmd_dry_run(fixtures_dir, out_dir) == 0

    fixtures = materialize_output_fixtures(fixtures_dir)
    cases = build_eval_cases()

    s_stub = summarize_grades(
        "tools_stub", "offline", run_offline_arm("tools_stub", cases, fixtures)
    )
    s_acc = summarize_grades(
        "decoy_always_accept", "offline", run_offline_arm("decoy_always_accept", cases, fixtures)
    )
    s_rej = summarize_grades(
        "decoy_always_reject", "offline", run_offline_arm("decoy_always_reject", cases, fixtures)
    )

    assert s_stub.overall_accuracy == 1.0
    assert s_stub.grounded_rate == 1.0
    assert s_acc.trap_caught_rate == 0.0 and s_acc.false_alarm_rate == 0.0
    assert s_rej.trap_caught_rate == 1.0 and s_rej.false_alarm_rate == 1.0
