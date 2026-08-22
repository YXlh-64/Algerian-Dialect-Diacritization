from pathlib import Path

import pytest

from experiments.track4.Lyes.dual_rope_boundary_crf_v8_ensemble import _existing_paths


def test_existing_paths_preserves_controlled_group_order(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    assert _existing_paths(
        [str(first), str(second)], "test"
    ) == [first, second]


def test_existing_paths_fails_closed_on_partial_group(
    tmp_path: Path,
) -> None:
    present = tmp_path / "present.pt"
    present.write_bytes(b"present")
    with pytest.raises(FileNotFoundError, match="missing"):
        _existing_paths(
            [str(present), str(tmp_path / "missing.pt")], "test"
        )
