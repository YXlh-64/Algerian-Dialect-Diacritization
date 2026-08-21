"""Caching, training, and evaluation runtime for FilteredWordLattice-v14."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.track4.Lyes.gated_fusion.config import GatedFusionConfig
from utils.track4.Lyes.gated_fusion.fusion import (
    GatedFusionStatistics,
    apply_gated_fallback,
)
from experiments.track4.Lyes.campaign.diagnostics import (
    prediction_diagnostics,
    training_word_types,
)
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import (
    BatchCollator,
    CharacterDataset,
    SentenceRecord,
    validate_vocabulary_coverage,
)
from utils.track4.Lyes.labels import NUM_LABELS
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from models.track4.Lyes.dual_stream_crf_head import LinearChainCRF
from evaluation.track4.Lyes.paper_metrics import compute_paper_metrics
from utils.track4.Lyes.utils import sha256_file, write_json
from models.track4.Lyes.word_lattice import (
    WordCandidateTransformer,
    WordLattice,
    build_word_lattice,
    lattice_marginals,
    lattice_viterbi,
)


@dataclass(frozen=True)
class CachedLatticeBundle:
    checkpoint_path: str
    checkpoint_sha256: str
    data_sha256: str
    k: int
    contexts: Tuple[torch.Tensor, ...]
    lattices: Tuple[WordLattice, ...]
    base_log_probabilities: Tuple[torch.Tensor, ...]
    transitions: torch.Tensor
    start_transitions: torch.Tensor
    end_transitions: torch.Tensor

    def validate(self, records: Sequence[SentenceRecord]) -> None:
        if len(records) != len(self.lattices):
            raise ValueError("cached lattice record count mismatch")
        if len(self.contexts) != len(records):
            raise ValueError("cached lattice context count mismatch")
        if len(self.base_log_probabilities) != len(records):
            raise ValueError("cached lattice probability count mismatch")
        if self.k not in (4, 8):
            raise ValueError("cached lattice K must be 4 or 8")
        for record, context, lattice, probabilities in zip(
            records,
            self.contexts,
            self.lattices,
            self.base_log_probabilities,
        ):
            if record.sent_id != lattice.sent_id:
                raise ValueError("cached lattice sentence alignment failure")
            if context.shape != (len(record.chars), 256):
                raise ValueError("cached context shape mismatch")
            if probabilities.shape != (len(record.chars), NUM_LABELS):
                raise ValueError("cached probability shape mismatch")
            if not torch.isfinite(context).all():
                raise ValueError("cached contexts must be finite")
            if not torch.isfinite(probabilities).all():
                raise ValueError("cached probabilities must be finite")


@torch.inference_mode()
def build_cached_bundle(
    checkpoint_path: Path,
    data_path: Path,
    records: Sequence[SentenceRecord],
    k: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> CachedLatticeBundle:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    if model.config.resolved_head_mode != "crf" or model.crf is None:
        raise ValueError("v14 requires a standard CRF checkpoint")
    if model.config.d_model != 256:
        raise ValueError("v14 is locked to a 256-dimensional v7 encoder")
    validate_vocabulary_coverage(records, vocab)
    loader = DataLoader(
        CharacterDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BatchCollator(vocab),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    cpu_crf = LinearChainCRF(NUM_LABELS)
    with torch.no_grad():
        cpu_crf.transitions.copy_(model.crf.transitions.detach().cpu())
    contexts: List[torch.Tensor] = []
    lattices: List[WordLattice] = []
    probabilities: List[torch.Tensor] = []
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        hidden, _ = model.encode(input_ids, attention_mask)
        if model.label_head is None:
            raise RuntimeError("v7 label head is missing")
        logits = model.label_head(hidden)
        outputs = {
            "logits": logits,
            "crf_mask": (
                attention_mask
                & input_ids.ne(model.config.space_id)
                & input_ids.ne(model.config.bos_id)
                & input_ids.ne(model.config.eos_id)
            ),
        }
        decoded = model.decode_outputs(outputs).cpu()
        log_probabilities = model.log_probabilities(outputs).cpu()
        hidden = hidden.cpu()
        logits = logits.cpu()
        for row, record in enumerate(batch["records"]):
            record_slice = slice(1, len(record.chars) + 1)
            record_context = hidden[row, record_slice].contiguous()
            baseline = decoded[row, record_slice].tolist()
            baseline = [
                0 if char == " " else int(label)
                for char, label in zip(record.chars, baseline)
            ]
            contexts.append(record_context)
            probabilities.append(
                log_probabilities[row, record_slice].contiguous()
            )
            lattices.append(
                build_word_lattice(
                    record,
                    logits[row, record_slice],
                    baseline,
                    cpu_crf,
                    k,
                )
            )
    bundle = CachedLatticeBundle(
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
        data_sha256=sha256_file(data_path),
        k=k,
        contexts=tuple(contexts),
        lattices=tuple(lattices),
        base_log_probabilities=tuple(probabilities),
        transitions=model.crf.transitions.detach().cpu(),
        start_transitions=model.crf.start_transitions.detach().cpu(),
        end_transitions=model.crf.end_transitions.detach().cpu(),
    )
    bundle.validate(records)
    return bundle


def load_or_build_cached_bundle(
    cache_path: Path,
    checkpoint_path: Path,
    data_path: Path,
    records: Sequence[SentenceRecord],
    k: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> CachedLatticeBundle:
    expected_checkpoint_hash = sha256_file(checkpoint_path)
    expected_data_hash = sha256_file(data_path)
    if cache_path.is_file():
        bundle = torch.load(
            cache_path, map_location="cpu", weights_only=False
        )
        if not isinstance(bundle, CachedLatticeBundle):
            raise RuntimeError("invalid v14 cache object")
        if (
            bundle.checkpoint_sha256 != expected_checkpoint_hash
            or bundle.data_sha256 != expected_data_hash
            or bundle.k != k
        ):
            raise RuntimeError("v14 cache evidence mismatch")
        bundle.validate(records)
        return bundle
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_cached_bundle(
        checkpoint_path,
        data_path,
        records,
        k,
        device,
        batch_size,
        num_workers,
    )
    torch.save(bundle, cache_path)
    write_json(
        cache_path.with_suffix(".manifest.json"),
        {
            "schema_version": 1,
            "cache_path": str(cache_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": expected_checkpoint_hash,
            "data_path": str(data_path),
            "data_sha256": expected_data_hash,
            "k": k,
            "records": len(records),
        },
    )
    return bundle


@dataclass(frozen=True)
class WordTrainingExample:
    context: torch.Tensor
    candidate_labels: Tuple[Tuple[int, ...], ...]
    base_scores: Tuple[float, ...]
    gold_index: int


class WordCandidateDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SentenceRecord],
        bundle: CachedLatticeBundle,
    ) -> None:
        bundle.validate(records)
        examples: List[WordTrainingExample] = []
        uncovered_words = 0
        total_words = 0
        for record, context, lattice in zip(
            records, bundle.contexts, bundle.lattices
        ):
            if record.labels is None:
                raise ValueError("v14 training records require labels")
            for (start, end), group in zip(
                lattice.spans, lattice.candidates
            ):
                total_words += 1
                gold = tuple(int(label) for label in record.labels[start:end])
                matching = [
                    index
                    for index, candidate in enumerate(group)
                    if candidate.labels == gold
                ]
                if not matching:
                    uncovered_words += 1
                    continue
                examples.append(
                    WordTrainingExample(
                        context=context[start:end].clone(),
                        candidate_labels=tuple(
                            candidate.labels for candidate in group
                        ),
                        base_scores=tuple(
                            float(candidate.base_score) for candidate in group
                        ),
                        gold_index=matching[0],
                    )
                )
        if not examples:
            raise ValueError("v14 training contains no covered gold words")
        self.examples = examples
        self.total_words = total_words
        self.uncovered_words = uncovered_words

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> WordTrainingExample:
        return self.examples[index]


def collate_word_examples(
    examples: Sequence[WordTrainingExample],
) -> Mapping[str, torch.Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty word batch")
    maximum_candidates = max(
        len(example.candidate_labels) for example in examples
    )
    maximum_length = max(example.context.size(0) for example in examples)
    context_dim = int(examples[0].context.size(1))
    batch_size = len(examples)
    context = torch.zeros(
        batch_size,
        maximum_candidates,
        maximum_length,
        context_dim,
    )
    labels = torch.zeros(
        batch_size, maximum_candidates, maximum_length, dtype=torch.long
    )
    letter_mask = torch.zeros(
        batch_size, maximum_candidates, maximum_length, dtype=torch.bool
    )
    candidate_mask = torch.zeros(
        batch_size, maximum_candidates, dtype=torch.bool
    )
    base_scores = torch.zeros(batch_size, maximum_candidates)
    gold_indices = torch.empty(batch_size, dtype=torch.long)
    for row, example in enumerate(examples):
        length = int(example.context.size(0))
        for candidate_index, candidate_labels in enumerate(
            example.candidate_labels
        ):
            context[row, candidate_index, :length] = example.context
            labels[row, candidate_index, :length] = torch.tensor(
                candidate_labels, dtype=torch.long
            )
            letter_mask[row, candidate_index, :length] = True
            candidate_mask[row, candidate_index] = True
            base_scores[row, candidate_index] = example.base_scores[
                candidate_index
            ]
        gold_indices[row] = example.gold_index
    return {
        "context": context,
        "labels": labels,
        "letter_mask": letter_mask,
        "candidate_mask": candidate_mask,
        "base_scores": base_scores,
        "gold_indices": gold_indices,
    }


def candidate_logits(
    scorer: WordCandidateTransformer,
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    context = batch["context"].to(device)
    labels = batch["labels"].to(device)
    letter_mask = batch["letter_mask"].to(device)
    candidate_mask = batch["candidate_mask"].to(device)
    base_scores = batch["base_scores"].to(device)
    batch_size, candidate_count, length, context_dim = context.shape
    residuals = scorer(
        context.reshape(batch_size * candidate_count, length, context_dim),
        labels.reshape(batch_size * candidate_count, length),
        letter_mask.reshape(batch_size * candidate_count, length),
    ).reshape(batch_size, candidate_count)
    return (base_scores + residuals).masked_fill(
        ~candidate_mask, torch.finfo(base_scores.dtype).min
    )


@torch.inference_mode()
def score_lattice_candidates(
    scorer: WordCandidateTransformer,
    bundle: CachedLatticeBundle,
    device: torch.device,
    batch_size_words: int = 256,
) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    """Score all candidate words in contiguous batches on the target device."""

    if batch_size_words <= 0:
        raise ValueError("candidate scoring batch size must be positive")
    examples: List[WordTrainingExample] = []
    word_counts: List[int] = []
    candidate_counts: List[int] = []
    for context, lattice in zip(bundle.contexts, bundle.lattices):
        word_counts.append(len(lattice.spans))
        for (start, end), group in zip(lattice.spans, lattice.candidates):
            examples.append(
                WordTrainingExample(
                    context=context[start:end].clone(),
                    candidate_labels=tuple(
                        candidate.labels for candidate in group
                    ),
                    base_scores=tuple(
                        float(candidate.base_score) for candidate in group
                    ),
                    gold_index=0,
                )
            )
            candidate_counts.append(len(group))
    loader = DataLoader(
        examples,
        batch_size=batch_size_words,
        shuffle=False,
        collate_fn=collate_word_examples,
        num_workers=0,
    )
    scorer.eval()
    flat_scores: List[torch.Tensor] = []
    offset = 0
    for batch in loader:
        scores = candidate_logits(scorer, batch, device).cpu()
        for row in range(scores.size(0)):
            flat_scores.append(
                scores[row, : candidate_counts[offset]].contiguous()
            )
            offset += 1
    if offset != len(candidate_counts):
        raise RuntimeError("candidate score count mismatch")
    nested: List[Tuple[torch.Tensor, ...]] = []
    offset = 0
    for count in word_counts:
        nested.append(tuple(flat_scores[offset : offset + count]))
        offset += count
    return tuple(nested)


@dataclass(frozen=True)
class LatticeEvaluation:
    metrics: Mapping[str, Any]
    neural_predictions: Tuple[Tuple[int, ...], ...]
    v2_predictions: Tuple[Tuple[int, ...], ...]
    probabilities: Tuple[torch.Tensor, ...]


def _flatten_metrics(
    paper: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "correct": int(paper["correct_letters"]),
        "total": int(paper["scored_letters"]),
        "micro_f1": float(paper["micro_f1"]),
        "macro_f1": float(paper["macro_f1"]),
        "word_accuracy": float(paper["word_accuracy"]),
        "word_correct": int(paper["word_correct"]),
        "sentence_accuracy": float(paper["sentence_accuracy"]),
        "sentence_correct": int(paper["sentence_correct"]),
        "oov_accuracy": float(diagnostics["oov_accuracy"]),
        "seen_accuracy": float(diagnostics["seen_accuracy"]),
        "shadda_accuracy": float(paper["shadda"]["accuracy"]),
        "tanween_accuracy": float(paper["tanween"]["accuracy"]),
        "skeleton_mismatch_count": int(paper["skeleton_mismatch_count"]),
        "paper_metrics": dict(paper),
        "diagnostics": dict(diagnostics),
    }


@torch.inference_mode()
def evaluate_lattice_scorer(
    scorer: WordCandidateTransformer,
    bundle: CachedLatticeBundle,
    train_records: Sequence[SentenceRecord],
    records: Sequence[SentenceRecord],
    gates: GatedFusionConfig,
    device: torch.device,
) -> LatticeEvaluation:
    bundle.validate(records)
    scorer.eval()
    prior = WordLabelPrior().fit(train_records)
    seen_words = training_word_types(train_records)
    transitions = bundle.transitions
    starts = bundle.start_transitions
    ends = bundle.end_transitions
    all_word_scores = score_lattice_candidates(scorer, bundle, device)
    neural_predictions: List[Tuple[int, ...]] = []
    v2_predictions: List[Tuple[int, ...]] = []
    probabilities: List[torch.Tensor] = []
    gate_statistics = GatedFusionStatistics()
    for record, lattice, word_scores in zip(
        records, bundle.lattices, all_word_scores
    ):
        neural = lattice_viterbi(
            lattice, word_scores, transitions, starts, ends
        )
        marginal = lattice_marginals(
            lattice, word_scores, transitions, starts, ends
        ).cpu()
        v2, statistics = apply_gated_fallback(
            record,
            marginal.clamp_min(1.0e-12).log(),
            prior,
            gates,
            initial_predictions=torch.tensor(neural),
        )
        gate_statistics.update(statistics)
        neural_predictions.append(tuple(neural))
        v2_predictions.append(tuple(v2))
        probabilities.append(marginal)
    neural_lists = [list(values) for values in neural_predictions]
    v2_lists = [list(values) for values in v2_predictions]
    if any(record.labels is None for record in records):
        return LatticeEvaluation(
            metrics={
                "v2_gate_statistics": gate_statistics.to_dict(),
                "evaluation_skipped": "records_have_no_gold_labels",
            },
            neural_predictions=tuple(neural_predictions),
            v2_predictions=tuple(v2_predictions),
            probabilities=tuple(probabilities),
        )
    neural_paper = compute_paper_metrics(records, neural_lists)
    v2_paper = compute_paper_metrics(records, v2_lists)
    metrics = {
        "neural": _flatten_metrics(
            neural_paper,
            prediction_diagnostics(records, neural_lists, seen_words),
        ),
        "v2": _flatten_metrics(
            v2_paper,
            prediction_diagnostics(records, v2_lists, seen_words),
        ),
        "v2_gate_statistics": gate_statistics.to_dict(),
    }
    return LatticeEvaluation(
        metrics=metrics,
        neural_predictions=tuple(neural_predictions),
        v2_predictions=tuple(v2_predictions),
        probabilities=tuple(probabilities),
    )


def evaluate_baseline_bundle(
    bundle: CachedLatticeBundle,
    train_records: Sequence[SentenceRecord],
    records: Sequence[SentenceRecord],
    gates: GatedFusionConfig,
) -> LatticeEvaluation:
    bundle.validate(records)
    prior = WordLabelPrior().fit(train_records)
    seen_words = training_word_types(train_records)
    neural_predictions = tuple(
        tuple(lattice.baseline_labels) for lattice in bundle.lattices
    )
    v2_predictions: List[Tuple[int, ...]] = []
    statistics = GatedFusionStatistics()
    for record, labels, log_probabilities in zip(
        records,
        neural_predictions,
        bundle.base_log_probabilities,
    ):
        v2, row_statistics = apply_gated_fallback(
            record,
            log_probabilities,
            prior,
            gates,
            initial_predictions=torch.tensor(labels),
        )
        v2_predictions.append(tuple(v2))
        statistics.update(row_statistics)
    neural_lists = [list(values) for values in neural_predictions]
    v2_lists = [list(values) for values in v2_predictions]
    neural_paper = compute_paper_metrics(records, neural_lists)
    v2_paper = compute_paper_metrics(records, v2_lists)
    metrics = {
        "neural": _flatten_metrics(
            neural_paper,
            prediction_diagnostics(records, neural_lists, seen_words),
        ),
        "v2": _flatten_metrics(
            v2_paper,
            prediction_diagnostics(records, v2_lists, seen_words),
        ),
        "v2_gate_statistics": statistics.to_dict(),
    }
    return LatticeEvaluation(
        metrics=metrics,
        neural_predictions=neural_predictions,
        v2_predictions=tuple(v2_predictions),
        probabilities=tuple(
            values.exp() for values in bundle.base_log_probabilities
        ),
    )


def build_scorer(config: Mapping[str, Any]) -> WordCandidateTransformer:
    return WordCandidateTransformer(
        context_dim=int(config["context_dim"]),
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        ffn_dim=int(config["ffn_dim"]),
        dropout=float(config["dropout"]),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_lattice_scorer(
    train_records: Sequence[SentenceRecord],
    train_bundle: CachedLatticeBundle,
    evaluation_records: Sequence[SentenceRecord],
    evaluation_bundle: CachedLatticeBundle,
    gates: GatedFusionConfig,
    scorer_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    fixed_epochs: Optional[int] = None,
) -> Mapping[str, Any]:
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint_path = Path(summary["best_checkpoint"])
        if not checkpoint_path.is_file():
            raise RuntimeError("partial v14 scorer run")
        return summary
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            "partial v14 scorer artifacts require manual inspection: {}".format(
                output_dir
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(training_config["seed"])
    _seed_everything(seed)
    dataset = WordCandidateDataset(train_records, train_bundle)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size_words"]),
        shuffle=True,
        collate_fn=collate_word_examples,
        generator=generator,
        num_workers=0,
    )
    scorer = build_scorer(scorer_config).to(device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = (
        int(fixed_epochs)
        if fixed_epochs is not None
        else int(training_config["epochs"])
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(training_config["learning_rate"]),
        total_steps=epochs * len(loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0,
    )
    metrics_path = output_dir / "metrics.jsonl"
    best_key: Optional[Tuple[int, int, int]] = None
    best_epoch = 0
    best_metrics: Optional[Mapping[str, Any]] = None
    for epoch in range(1, epochs + 1):
        scorer.train()
        total_loss = 0.0
        examples = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = candidate_logits(scorer, batch, device)
            targets = batch["gold_indices"].to(device)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(
                scorer.parameters(),
                float(training_config["gradient_clip_norm"]),
            )
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu()) * targets.numel()
            examples += targets.numel()
        evaluation: Optional[LatticeEvaluation] = None
        if fixed_epochs is None or epoch == epochs:
            evaluation = evaluate_lattice_scorer(
                scorer,
                evaluation_bundle,
                train_records,
                evaluation_records,
                gates,
                device,
            )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / examples,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "dev": None if evaluation is None else evaluation.metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if evaluation is not None:
            neural = evaluation.metrics["neural"]
            key = (
                int(neural["correct"]),
                int(neural["word_correct"]),
                -epoch,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                best_metrics = evaluation.metrics
                torch.save(
                    {
                        "schema_version": 1,
                        "epoch": epoch,
                        "k": train_bundle.k,
                        "scorer_config": dict(scorer_config),
                        "training_config": dict(training_config),
                        "scorer_state_dict": scorer.state_dict(),
                        "base_checkpoint": train_bundle.checkpoint_path,
                        "base_checkpoint_sha256": train_bundle.checkpoint_sha256,
                        "dev_metrics": evaluation.metrics,
                    },
                    output_dir / "best.pt",
                )
        torch.save(
            {
                "schema_version": 1,
                "epoch": epoch,
                "k": train_bundle.k,
                "scorer_config": dict(scorer_config),
                "training_config": dict(training_config),
                "scorer_state_dict": scorer.state_dict(),
                "base_checkpoint": train_bundle.checkpoint_path,
                "base_checkpoint_sha256": train_bundle.checkpoint_sha256,
            },
            output_dir / "last.pt",
        )
    if best_metrics is None:
        raise RuntimeError("v14 scorer produced no evaluation")
    summary = {
        "schema_version": 1,
        "device": str(device),
        "k": train_bundle.k,
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "best_checkpoint": str(output_dir / "best.pt"),
        "best_metrics": best_metrics,
        "parameter_count": sum(p.numel() for p in scorer.parameters()),
        "training_words": len(dataset),
        "uncovered_training_words": dataset.uncovered_words,
        "total_training_words": dataset.total_words,
        "fixed_epoch_mode": fixed_epochs is not None,
    }
    write_json(summary_path, summary)
    return summary


def load_scorer_checkpoint(
    path: Path, device: torch.device
) -> Tuple[WordCandidateTransformer, Mapping[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported v14 scorer checkpoint")
    scorer = build_scorer(checkpoint["scorer_config"]).to(device)
    scorer.load_state_dict(checkpoint["scorer_state_dict"], strict=True)
    scorer.eval()
    return scorer, checkpoint
