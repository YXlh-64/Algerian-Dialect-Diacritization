# Shared evaluate.py for track3/linear_head (6 of 6 experiments — auto-consolidated, identical after stripping header/noise). Excluded (differ for real reasons): none

# ## 9. Final Local Evaluation on `DEV_TEST`
class Evaluator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    @torch.no_grad()
    def evaluate(self, models: List[nn.Module], loader,
                 predict_fn=None, word_metric_records: Optional[List[dict]] = None) -> Dict[str, Any]:
        for m in models:
            m.eval()
        y_true, y_pred = [], []
        for batch in loader:
            batch_gpu = {k: v.to(DEVICE) for k, v in batch.items()}
            probs_sum = None
            for m in models:
                logits = m(batch_gpu["input_ids"], batch_gpu["attention_mask"],
                           batch_gpu["char_ids"], batch_gpu["token_idx_per_char"])
                probs = torch.softmax(logits, dim=-1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            preds = (probs_sum / len(models)).argmax(-1).cpu()
            labels = batch["char_labels"]
            mask = labels != IGNORE_INDEX
            y_true.extend(labels[mask].tolist())
            y_pred.extend(preds[mask].tolist())

        n_classes = len(self.class_names)
        present_labels = sorted(set(y_true))

        micro_f1 = f1_score(y_true, y_pred, average="micro")
        macro_f1_all16 = f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)))
        macro_f1_present = f1_score(y_true, y_pred, average="macro", labels=present_labels)

        report = classification_report(y_true, y_pred, target_names=self.class_names,
                                        labels=list(range(n_classes)),
                                        output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

        result = {
            "micro_f1": micro_f1,
            "macro_f1_all16": macro_f1_all16,               # kept for continuity with earlier exports
            "macro_f1_present_classes": macro_f1_present,    # honest number: excludes 0-support classes
            "n_present_classes": len(present_labels),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "n_chars_evaluated": len(y_true),
            "n_models_ensembled": len(models),
        }
        if predict_fn is not None and word_metric_records is not None:
            result.update(word_level_metrics_from_predict_fn(predict_fn, word_metric_records))
        return result

    def plot_confusion(self, cm: List[List[int]], normalize: bool = True):
        cm = np.array(cm, dtype=float)
        if normalize:
            cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(self.class_names))); ax.set_xticklabels(self.class_names, rotation=90)
        ax.set_yticks(range(len(self.class_names))); ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion matrix (row-normalized)")
        plt.colorbar(im, fraction=0.046)
        plt.tight_layout(); plt.show()


def print_eval_report(report: Dict[str, Any], class_names: List[str], name: str = "DEV_TEST") -> None:
    '''One readable printout pulling from the same report dict that gets
    saved/exported -- so the notebook output and the exported JSON never
    disagree again.'''
    print(f"=== {name} ===")
    print(f"Micro-F1 (competition metric)                : {report['micro_f1']:.4f}")
    print(f"Macro-F1 (all 16 classes, legacy/comparable)  : {report['macro_f1_all16']:.4f}")
    print(f"Macro-F1 ({report['n_present_classes']} classes actually present)        : "
          f"{report['macro_f1_present_classes']:.4f}")
    if "DER" in report:
        print(f"DER  / DER* (excl. word-final letter)        : {report['DER']:.4f} / {report['DER_star']:.4f}")
        print(f"WER  / WER* (excl. word-final letter)        : {report['WER']:.4f} / {report['WER_star']:.4f}")
        print(f"Sentence exact-match accuracy                : {report['sentence_exact_match']:.4f}")
    print(f"Characters evaluated                         : {report['n_chars_evaluated']}")
    print(f"Models ensembled                             : {report['n_models_ensembled']}")
    if "per_class_der" in report:
        print("\nPer-class DER (char error rate), worst first:")
        for cid, der in sorted(report["per_class_der"].items(), key=lambda x: -x[1]):
            print(f"  {class_names[cid]:20s} DER={der:.4f}")
    if "top_confusions" in report:
        print("\nTop confusions (true -> predicted : count):")
        for (t, p), n in report["top_confusions"]:
            print(f"  {class_names[t]:18s} -> {class_names[p]:18s} : {n}")
    print()


def build_fresh_model() -> Track3Diacritizer:
    return Track3Diacritizer(
        BACKBONE_NAME, len(CHAR2ID), CFG.num_classes, CFG.char_emb_dim,
        n_concat_layers=CFG.n_concat_layers, head_hidden_dim=CFG.head_hidden_dim,
        head_dropout=CFG.head_dropout, use_deep_head=CFG.use_deep_head,
    ).to(DEVICE)


if CFG.k_folds > 1 and FOLD_CHECKPOINT_DIRS:
    MODELS_FOR_INFERENCE = []
    for fold_dir in FOLD_CHECKPOINT_DIRS:
        fm = build_fresh_model()
        fold_ckpt = CheckpointManager(fold_dir)
        fold_ckpt.load_best(fm)
        fold_ckpt.free_latest()   # weights are safely loaded now -- drop the
                                   # full (model+optimizer+scheduler) latest.pt
                                   # before self-training starts writing more
        MODELS_FOR_INFERENCE.append(fm)
    print(f"Loaded {len(MODELS_FOR_INFERENCE)} fold models for ensemble evaluation.")
    report_disk_usage()
else:
    ckpt.load_best(model)   # best checkpoint by dev micro-F1, from Section 8
    ckpt.free_latest()
    MODELS_FOR_INFERENCE = [model]


@torch.no_grad()
def _predict_chars(models: List[nn.Module], chars: List[str]) -> List[int]:
    '''Ensembled argmax prediction per character for one sentence. Defined
    here (before the first evaluate() call) so it is available both for this
    DEV_TEST evaluation and for the self-training section further down.'''
    if not chars:
        return []
    enc = aligner.encode(chars)
    input_ids = torch.tensor([enc["input_ids"]], device=DEVICE)
    attn = torch.ones_like(input_ids)
    toks = torch.tensor([[t if t >= 0 else 0 for t in enc["token_idx_per_char"][:len(chars)]]],
                         device=DEVICE)
    char_ids = torch.tensor([[CHAR2ID.get(c, CHAR2ID.get('<UNK>', 1)) for c in chars]], device=DEVICE)
    probs_sum = None
    for m in models:
        logits = m(input_ids, attn, char_ids, toks)[0]
        probs = torch.softmax(logits, dim=-1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return (probs_sum / len(models)).argmax(-1).cpu().tolist()


dev_test_ds = DiacritizationDataset(dev_test_records, aligner, CHAR2ID)
dev_test_loader = DataLoader(dev_test_ds, batch_size=CFG.eval_batch_size, shuffle=False, collate_fn=_collate)

evaluator = Evaluator(CLASS_NAMES)
DEV_TEST_REPORT = evaluator.evaluate(
    MODELS_FOR_INFERENCE, dev_test_loader,
    predict_fn=lambda chars: _predict_chars(MODELS_FOR_INFERENCE, chars),
    word_metric_records=dev_test_records,
)
DEV_TEST_SCORE = DEV_TEST_REPORT["micro_f1"]

print_eval_report(DEV_TEST_REPORT, CLASS_NAMES, name="DEV_TEST")
evaluator.plot_confusion(DEV_TEST_REPORT["confusion_matrix"])
