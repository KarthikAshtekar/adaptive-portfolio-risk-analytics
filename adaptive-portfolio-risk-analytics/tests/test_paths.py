"""Tests for portable repository path resolution."""

from pathlib import Path

from src.config import ConfigManager
from src.paths import CONFIG_DIR, PROJECT_ROOT, project_path


def test_project_paths_are_absolute_and_repo_scoped() -> None:
    assert PROJECT_ROOT.is_absolute()
    assert CONFIG_DIR == PROJECT_ROOT / "config"
    assert project_path("outputs", "example.csv") == (PROJECT_ROOT / "outputs" / "example.csv")


def test_default_config_path_does_not_depend_on_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    manager = ConfigManager()

    assert manager.config_dir == CONFIG_DIR
    assert manager.get("portfolio.lookback_window") is not None
