"""Shared pytest fixtures for real, disposable SQLite databases."""

from pathlib import Path

import pytest

from scripts.create_demo_database import create_demo_database


@pytest.fixture
def demo_db(tmp_path: Path) -> Path:
    """Create a deterministic database isolated to one test."""

    return create_demo_database(tmp_path / "demo.db")
