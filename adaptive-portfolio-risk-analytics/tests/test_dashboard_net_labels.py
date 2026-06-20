"""Tests for explicit net/gross dashboard labeling."""

from src.dashboard.modes import NET_METRIC_LABELS, net_metric_label


def test_headline_return_metrics_have_explicit_net_labels() -> None:
    assert net_metric_label("cagr") == "Net CAGR"
    assert net_metric_label("sharpe") == "Net Sharpe"
    assert net_metric_label("sortino") == "Net Sortino"
    assert net_metric_label("calmar") == "Net Calmar"
    assert net_metric_label("final_value") == "Net Final Value"
    assert net_metric_label("excess_cagr") == "Net Excess CAGR"
    assert (
        net_metric_label("strategy_stress_return")
        == "Strategy Net Stress Return"
    )


def test_gross_is_not_used_as_a_headline_metric_label() -> None:
    assert all("Gross" not in label for label in NET_METRIC_LABELS.values())
