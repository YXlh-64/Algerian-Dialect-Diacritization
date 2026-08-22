import pytest

from utils.track4.Lyes.labels import (
    LABEL_MARKS,
    apply_diacritics,
    combine_label,
    labels_to_components,
    split_label,
)


def test_all_labels_round_trip_through_factorization() -> None:
    for label in range(16):
        base, shadda = split_label(label)
        assert combine_label(base, shadda) == label


def test_vector_factorization_preserves_order() -> None:
    base, shadda = labels_to_components(range(16))
    assert base == list(range(8)) + list(range(8))
    assert shadda == [0] * 8 + [1] * 8


def test_apply_diacritics_preserves_skeleton_and_marks() -> None:
    chars = tuple("ب ب")
    for label in range(16):
        labels = [label, 0, label]
        assert apply_diacritics(chars, labels) == (
            "ب" + LABEL_MARKS[label] + " " + "ب" + LABEL_MARKS[label]
        )


def test_space_must_have_zero_label() -> None:
    with pytest.raises(ValueError, match="space"):
        apply_diacritics(("ب", " "), (1, 1))
