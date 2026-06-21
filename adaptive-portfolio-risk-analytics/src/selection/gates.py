"""Evidence and safety gates used before a strategy can be recommended."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Mapping


class GateStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    reason: str
    value: object = None
    threshold: object = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


def _number(values: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        try:
            value = float(values.get(key))
        except (TypeError, ValueError):
            continue
        if isfinite(value):
            return value
    return None


def _text(values: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = values.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def evaluate_selection_gates(
    candidate: Mapping[str, object],
    *,
    cpcv: Mapping[str, object] | None = None,
    stress: Mapping[str, object] | None = None,
    minimum_observations: int = 504,
    high_cost_bps: float = 50.0,
) -> list[GateResult]:
    """Evaluate a candidate using only declared net, out-of-sample evidence."""

    cpcv = cpcv or {}
    stress = stress or {}
    results: list[GateResult] = []
    strategy_type = _text(candidate, "strategy_type").lower()
    regime_source = _text(candidate, "regime_source").lower()

    return_basis = _text(candidate, "return_basis").lower()
    if return_basis == "net":
        results.append(GateResult("Net return basis", GateStatus.PASS, "Metrics are net of costs.", return_basis))
    elif return_basis:
        results.append(GateResult("Net return basis", GateStatus.FAIL, "Gross-only evidence cannot support selection.", return_basis, "net"))
    else:
        results.append(GateResult("Net return basis", GateStatus.WARN, "Return basis was not explicitly labelled net.", None, "net"))

    full_sample_hmm = "hmm" in regime_source and "walk_forward" not in regime_source and "walk-forward" not in regime_source
    if full_sample_hmm:
        results.append(GateResult("No full-sample HMM", GateStatus.FAIL, "Full-sample HMM is historical diagnostics only.", regime_source, "walk-forward"))
    elif "hmm" in regime_source:
        results.append(GateResult("No full-sample HMM", GateStatus.PASS, "HMM evidence is walk-forward.", regime_source))
    else:
        results.append(GateResult("No full-sample HMM", GateStatus.NOT_AVAILABLE, "Candidate does not use HMM."))

    observations = _number(candidate, "n_observations", "observations")
    if observations is None:
        results.append(GateResult("Sufficient data", GateStatus.WARN, "Observation count is unavailable.", None, minimum_observations))
    elif observations >= minimum_observations:
        results.append(GateResult("Sufficient data", GateStatus.PASS, "History meets the minimum evidence window.", int(observations), minimum_observations))
    else:
        results.append(GateResult("Sufficient data", GateStatus.FAIL, "History is too short for this selection.", int(observations), minimum_observations))

    successful = _number(cpcv, "successful_folds")
    failed = _number(cpcv, "failed_folds")
    if successful is None:
        results.append(GateResult("CPCV coverage", GateStatus.NOT_AVAILABLE, "No CPCV result is available for this candidate."))
    else:
        total = successful + (failed or 0.0)
        coverage = successful / total if total else 0.0
        if coverage >= 0.60:
            status = GateStatus.PASS
        elif coverage >= 0.20:
            status = GateStatus.WARN
        else:
            status = GateStatus.FAIL
        results.append(GateResult("CPCV coverage", status, "Successful CPCV folds divided by attempted folds.", coverage, 0.60))

    worst_fold = _number(cpcv, "objective_worst")
    if worst_fold is None:
        results.append(GateResult("CPCV worst fold", GateStatus.NOT_AVAILABLE, "Worst-fold objective is unavailable."))
    elif worst_fold > 0:
        results.append(GateResult("CPCV worst fold", GateStatus.PASS, "Worst successful fold remains positive.", worst_fold, 0.0))
    else:
        results.append(GateResult("CPCV worst fold", GateStatus.WARN, "Worst successful fold is non-positive.", worst_fold, 0.0))

    turnover = _number(candidate, "total_turnover", "turnover", "average_turnover")
    if turnover is None:
        results.append(GateResult("Turnover", GateStatus.NOT_AVAILABLE, "Turnover evidence is unavailable."))
    elif turnover <= 12:
        results.append(GateResult("Turnover", GateStatus.PASS, "Turnover is within the lower implementation range.", turnover, 12.0))
    elif turnover <= 25:
        results.append(GateResult("Turnover", GateStatus.WARN, "Turnover is elevated and cost-sensitive.", turnover, 12.0))
    else:
        results.append(GateResult("Turnover", GateStatus.FAIL, "Turnover is outside the preferred implementation range.", turnover, 25.0))

    total_cost_bps = _number(candidate, "total_cost_bps", "cost_bps")
    if total_cost_bps is None:
        results.append(GateResult("Cost assumption", GateStatus.NOT_AVAILABLE, "Configured trading-cost assumption is unavailable."))
    elif total_cost_bps >= high_cost_bps:
        results.append(GateResult("Cost assumption", GateStatus.WARN, "High trading costs reduce confidence in active overlays.", total_cost_bps, high_cost_bps))
    else:
        results.append(GateResult("Cost assumption", GateStatus.PASS, "Trading-cost assumption is within the tested range.", total_cost_bps, high_cost_bps))

    stress_return = _number(stress, "period_return", "stress_period_return")
    stress_drawdown = _number(stress, "max_drawdown")
    if stress_return is None and stress_drawdown is None:
        results.append(GateResult("Stress evidence", GateStatus.NOT_AVAILABLE, "No stress-period result is available."))
    elif (stress_return is not None and stress_return >= 0) or (stress_drawdown is not None and stress_drawdown > -0.15):
        results.append(GateResult("Stress evidence", GateStatus.PASS, "Available stress evidence shows material protection.", stress_return if stress_return is not None else stress_drawdown))
    else:
        results.append(GateResult("Stress evidence", GateStatus.WARN, "Stress evidence does not establish clear protection.", stress_return if stress_return is not None else stress_drawdown))

    if strategy_type == "regime_adaptive":
        defensive_source = _text(candidate, "defensive_source_used", "defensive_source")
        if defensive_source:
            results.append(GateResult("Defensive metadata", GateStatus.PASS, "Defensive return source is recorded.", defensive_source))
        else:
            results.append(GateResult("Defensive metadata", GateStatus.WARN, "Defensive return source is not recorded."))
    else:
        results.append(GateResult("Defensive metadata", GateStatus.NOT_AVAILABLE, "Fixed strategy has no defensive sleeve."))

    if "hmm" in regime_source:
        hmm_valid = candidate.get("hmm_walk_forward_valid", candidate.get("status", "success") == "success")
        if bool(hmm_valid):
            results.append(GateResult("HMM walk-forward valid", GateStatus.PASS, "Walk-forward HMM completed successfully.", True))
        else:
            results.append(GateResult("HMM walk-forward valid", GateStatus.FAIL, "Walk-forward HMM is unavailable or unstable.", False))
    else:
        results.append(GateResult("HMM walk-forward valid", GateStatus.NOT_AVAILABLE, "Candidate does not use HMM."))

    classification = _text(candidate, "classification").lower()
    if classification:
        status = GateStatus.PASS if classification in {"risk-control overlay", "main growth strategy"} else GateStatus.WARN
        results.append(GateResult("Replication classification", status, f"Replication classified the strategy as {classification}.", classification))
    elif strategy_type == "regime_adaptive":
        results.append(GateResult("Replication classification", GateStatus.NOT_AVAILABLE, "Replication classification is unavailable."))

    return results


def gate_summary(gates: list[GateResult]) -> dict[str, int]:
    """Count gate outcomes for compact confidence calculations."""

    return {
        status.value: sum(gate.status == status for gate in gates)
        for status in GateStatus
    }

