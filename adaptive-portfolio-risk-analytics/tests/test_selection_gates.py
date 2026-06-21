"""Unit tests for selection safety and evidence gates."""

from __future__ import annotations

from src.selection.gates import GateStatus, evaluate_selection_gates


def _statuses(gates):
    return {gate.gate: gate.status for gate in gates}


def test_gross_evidence_and_full_sample_hmm_fail() -> None:
    gates = evaluate_selection_gates(
        {
            "strategy_type": "regime_adaptive",
            "return_basis": "gross",
            "regime_source": "hmm_full_sample",
            "n_observations": 1000,
            "total_turnover": 8,
            "total_cost_bps": 15,
            "defensive_source_used": "synthetic",
        }
    )
    statuses = _statuses(gates)

    assert statuses["Net return basis"] == GateStatus.FAIL
    assert statuses["No full-sample HMM"] == GateStatus.FAIL


def test_walk_forward_hmm_and_net_basis_pass() -> None:
    gates = evaluate_selection_gates(
        {
            "strategy_type": "regime_adaptive",
            "return_basis": "net",
            "regime_source": "hmm_walk_forward",
            "n_observations": 1000,
            "total_turnover": 8,
            "total_cost_bps": 15,
            "defensive_source_used": "synthetic",
            "hmm_walk_forward_valid": True,
        },
        cpcv={
            "successful_folds": 10,
            "failed_folds": 5,
            "objective_worst": 0.25,
        },
    )
    statuses = _statuses(gates)

    assert statuses["Net return basis"] == GateStatus.PASS
    assert statuses["No full-sample HMM"] == GateStatus.PASS
    assert statuses["HMM walk-forward valid"] == GateStatus.PASS
    assert statuses["CPCV coverage"] == GateStatus.PASS


def test_short_history_and_low_cpcv_coverage_fail() -> None:
    gates = evaluate_selection_gates(
        {
            "strategy_type": "fixed",
            "return_basis": "net",
            "n_observations": 200,
            "total_turnover": 3,
            "total_cost_bps": 15,
        },
        cpcv={
            "successful_folds": 1,
            "failed_folds": 9,
            "objective_worst": -0.1,
        },
    )
    statuses = _statuses(gates)

    assert statuses["Sufficient data"] == GateStatus.FAIL
    assert statuses["CPCV coverage"] == GateStatus.FAIL
    assert statuses["CPCV worst fold"] == GateStatus.WARN

