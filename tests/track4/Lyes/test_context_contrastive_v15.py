from pathlib import Path

import torch

from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from models.track4.Lyes.context_contrastive import (
    AmbiguityIndex,
    ContextContrastiveModel,
    pack_isolated_words,
    load_context_contrastive_checkpoint,
    save_context_contrastive_checkpoint,
)
from experiments.track4.Lyes.context_contrastive_v15 import (
    _make_loader,
    _reconcile_final_selection,
    load_campaign_config,
)
from utils.track4.Lyes.data import BatchCollator, SentenceRecord


CHECKPOINT = Path("outputs/dziriformer_dual_rope_crf_v7_seed42/best.pt")


def _base_and_vocab():
    checkpoint = load_checkpoint(CHECKPOINT, torch.device("cpu"))
    return build_model_from_checkpoint(checkpoint, torch.device("cpu"))


def test_isolated_word_packing_alignment() -> None:
    base, vocab = _base_and_vocab()
    records = [
        SentenceRecord("a", tuple("اب ج"), (1, 2, 0, 3), "اب ج"),
        SentenceRecord("b", tuple("د"), (4,), "د"),
    ]
    batch = BatchCollator(vocab)(records)
    packed = pack_isolated_words(
        batch["input_ids"],
        batch["attention_mask"],
        space_id=base.config.space_id,
        bos_id=base.config.bos_id,
        eos_id=base.config.eos_id,
        pad_id=base.config.pad_id,
    )
    assert packed.input_ids.size(0) == 3
    assert packed.sentence_indices.tolist() == [0, 0, 0, 1]
    assert packed.sentence_positions.tolist() == [1, 2, 4, 1]


def test_zero_residual_exactly_reproduces_v7_outputs() -> None:
    torch.manual_seed(7)
    base, vocab = _base_and_vocab()
    base.eval()
    model = ContextContrastiveModel(base).eval()
    records = [SentenceRecord("a", tuple("اب ج"), (1, 2, 0, 3), "اب ج")]
    batch = BatchCollator(vocab)(records)
    with torch.inference_mode():
        expected = base(batch["input_ids"], batch["attention_mask"])
        actual = model(batch["input_ids"], batch["attention_mask"])
    torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0.0, atol=0.0)
    assert torch.equal(model.decode_outputs(actual), base.decode_outputs(expected))


def test_shared_encoder_and_fusion_receive_gradients() -> None:
    torch.manual_seed(11)
    base, vocab = _base_and_vocab()
    model = ContextContrastiveModel(base)
    records = [SentenceRecord("a", tuple("اب اب"), (1, 2, 0, 1, 3), "اب اب")]
    batch = BatchCollator(vocab)(records)
    ambiguity = AmbiguityIndex.fit(records)
    outputs = model(batch["input_ids"], batch["attention_mask"])
    targets = ambiguity.targets(records, batch["input_ids"].size(1), torch.device("cpu"))
    loss, _, _ = model.compute_loss(outputs, batch["targets"], targets, 0.3)
    loss.backward()
    assert model.base.token_embedding.weight.grad is not None
    assert model.residual_projection.weight.grad is not None
    assert model.gate_network[0].weight.grad is not None


def test_ambiguity_targets_are_training_only_position_variants() -> None:
    records = [
        SentenceRecord("a", tuple("اب"), (1, 2), "اب"),
        SentenceRecord("b", tuple("اب"), (1, 3), "اب"),
    ]
    index = AmbiguityIndex.fit(records)
    targets = index.targets(records[:1], 4, torch.device("cpu"))
    assert targets[0, 1].item() == 0
    assert targets[0, 2].item() == 1


def test_context_checkpoint_round_trip(tmp_path: Path) -> None:
    base, vocab = _base_and_vocab()
    model = ContextContrastiveModel(base).eval()
    path = tmp_path / "best.pt"
    save_context_contrastive_checkpoint(path, model, vocab, 256, 3, 0.3, {"correct": 1})
    restored, restored_vocab, metadata = load_context_contrastive_checkpoint(path, torch.device("cpu"))
    assert restored_vocab == vocab
    assert metadata["epoch"] == 3
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_v15_campaign_config_is_locked() -> None:
    config = load_campaign_config(Path("configs/track4/Lyes/context_contrastive_v15/campaign.json"))
    assert config["auxiliary_coefficients"] == [0.1, 0.3, 1.0]
    assert config["training"]["gradient_accumulation_steps"] == 2


def test_v15_evaluation_loader_preserves_record_order() -> None:
    _, vocab = _base_and_vocab()
    records = [
        SentenceRecord("long", tuple("ابتث"), (0, 0, 0, 0), "ابتث"),
        SentenceRecord("short", tuple("ا"), (0,), "ا"),
        SentenceRecord("middle", tuple("اب"), (0, 0), "اب"),
    ]
    loader, _ = _make_loader(records, vocab, 2, False, 42, 0)
    observed = [record.sent_id for batch in loader for record in batch["records"]]
    assert observed == [record.sent_id for record in records]


def test_completed_v15_selection_is_reconciled_with_both_protected_variants() -> None:
    metrics = {
        "correct": 100,
        "word_accuracy": 0.8,
        "sentence_accuracy": 0.4,
        "oov_accuracy": 0.7,
        "shadda_accuracy": 0.9,
        "tanween_accuracy": 1.0,
        "skeleton_mismatch_count": 0,
    }
    selection = {
        "candidate": {
            "neural": {**metrics, "correct": 120},
            "v2": {**metrics, "correct": 115, "shadda_accuracy": 0.89},
        },
        "control": {"neural": metrics, "v2": metrics},
        "standalone_manifest": {"artifact_prefix": "TEST", "accepted": True},
        "ensemble_accepted": True,
        "accepted": True,
    }
    gate = {"minimum_neural_correct_gain": 15, "minimum_v2_correct_gain": 10}
    reconciled = _reconcile_final_selection(selection, gate)
    assert reconciled["protected_neural_regressions"]["shadda_accuracy"] is False
    assert reconciled["protected_v2_regressions"]["shadda_accuracy"] is True
    assert reconciled["standalone_accepted"] is False
    assert reconciled["standalone_manifest"]["accepted"] is False
    assert reconciled["accepted"] is False
