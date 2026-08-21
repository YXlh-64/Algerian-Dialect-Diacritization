"""Deterministic baselines and a competition-valid lexical submission."""

import argparse
import collections
import json
from pathlib import Path
from typing import Counter, DefaultDict, Dict, List, Sequence, Tuple

import torch

from utils.track4.Lyes.data import SentenceRecord, load_jsonl, load_raw_sentences
from evaluation.track4.Lyes.metrics import MetricAccumulator
from evaluation.track4.Lyes.submission import write_submission, write_vocalized_predictions


LabelSequence = Tuple[int, ...]


def _deterministic_mode(counter: Counter) -> object:
    if not counter:
        raise ValueError("cannot take the mode of an empty counter")
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


class LexiconBaseline:
    """Most frequent training vocalization per word with character fallback."""

    def __init__(self) -> None:
        self.character_labels: Dict[str, int] = {}
        self.word_labels: Dict[str, LabelSequence] = {}

    def fit(self, records: Sequence[SentenceRecord]) -> "LexiconBaseline":
        character_counts: DefaultDict[str, Counter] = collections.defaultdict(
            collections.Counter
        )
        word_counts: DefaultDict[str, Counter] = collections.defaultdict(
            collections.Counter
        )

        for record in records:
            if record.labels is None:
                raise ValueError("training records must contain labels")
            for char, label in zip(record.chars, record.labels):
                if char != " ":
                    character_counts[char][label] += 1
            start = 0
            for end in range(len(record.chars) + 1):
                if end == len(record.chars) or record.chars[end] == " ":
                    if end > start:
                        word = "".join(record.chars[start:end])
                        labels = tuple(record.labels[start:end])
                        word_counts[word][labels] += 1
                    start = end + 1

        self.character_labels = {
            char: int(_deterministic_mode(counts))
            for char, counts in character_counts.items()
        }
        self.word_labels = {
            word: tuple(_deterministic_mode(counts))
            for word, counts in word_counts.items()
        }
        return self

    def predict_record(self, record: SentenceRecord) -> List[int]:
        if not self.character_labels:
            raise RuntimeError("baseline must be fitted before prediction")
        labels: List[int] = []
        start = 0
        for end in range(len(record.chars) + 1):
            if end == len(record.chars) or record.chars[end] == " ":
                if end > start:
                    word = "".join(record.chars[start:end])
                    known = self.word_labels.get(word)
                    if known is not None:
                        labels.extend(known)
                    else:
                        for char in record.chars[start:end]:
                            if char not in self.character_labels:
                                labels.append(0)
                            else:
                                labels.append(self.character_labels[char])
                if end < len(record.chars):
                    labels.append(0)
                start = end + 1
        if len(labels) != len(record.chars):
            raise RuntimeError("internal baseline alignment failure")
        return labels

    def predict(self, records: Sequence[SentenceRecord]) -> List[List[int]]:
        return [self.predict_record(record) for record in records]


def evaluate(
    records: Sequence[SentenceRecord], predictions: Sequence[Sequence[int]]
) -> Dict[str, object]:
    accumulator = MetricAccumulator()
    for record, predicted in zip(records, predictions):
        if record.labels is None:
            raise ValueError("evaluation records must contain labels")
        targets = torch.tensor(
            [
                label if char != " " else -100
                for char, label in zip(record.chars, record.labels)
            ],
            dtype=torch.long,
        )
        accumulator.update(torch.tensor(predicted), targets)
    return accumulator.compute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("Data/train_data/train_Algerian-DIAC.jsonl"),
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=Path("Data/dev_data/dev_Algerian-DIAC.jsonl"),
    )
    parser.add_argument("--test-input", type=Path)
    parser.add_argument("--test-ids", type=Path)
    parser.add_argument("--vocalized-output", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--sample-submission", type=Path)
    args = parser.parse_args()

    train_records = load_jsonl(args.train)
    dev_records = load_jsonl(args.dev)
    baseline = LexiconBaseline().fit(train_records)
    metrics = evaluate(dev_records, baseline.predict(dev_records))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    requested_test_output = (
        args.vocalized_output is not None or args.submission is not None
    )
    if requested_test_output:
        if args.test_input is None or args.test_ids is None:
            parser.error(
                "--test-input and --test-ids are required for test prediction"
            )
        test_records = load_raw_sentences(args.test_input, args.test_ids)
        test_predictions = baseline.predict(test_records)
        if args.vocalized_output is not None:
            write_vocalized_predictions(
                args.vocalized_output, test_records, test_predictions
            )
        if args.submission is not None:
            write_submission(
                args.submission,
                test_records,
                test_predictions,
                sample_submission_path=args.sample_submission,
            )


if __name__ == "__main__":
    main()
