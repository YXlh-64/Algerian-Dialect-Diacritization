import glob
import json
import os

REQUIRED_FILES = [
    "train_Algerian-DIAC.jsonl", "dev_Algerian-DIAC.jsonl", "vocab.json", "class_labels.txt",
    "raw_sentences_test.txt", "raw_sentences_test_ids.txt", "sample_submission.csv", "make_submission.py",
]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "configs", "track1", "soundous", "paths.json")


def _has_all(p):
    return os.path.isdir(p) and all(os.path.isfile(os.path.join(p, r)) for r in REQUIRED_FILES)


def _autodetect_data_root():
    candidates = [os.path.join(_REPO_ROOT, "data"), ".", ".."]
    if os.path.isdir("/kaggle/input"):
        for sub in os.listdir("/kaggle/input"):
            p = os.path.join("/kaggle/input", sub)
            if _has_all(p):
                return p
            for sub2 in glob.glob(os.path.join(p, "*")):
                if _has_all(sub2):
                    return sub2
    for c in candidates:
        if _has_all(c):
            return c
    return None


def resolve_paths(data_root=None, output_dir=None):
    if data_root is None and os.path.isfile(_DEFAULT_CONFIG):
        with open(_DEFAULT_CONFIG) as f:
            cfg = json.load(f)
        data_root = cfg.get("data_root") or None
        output_dir = output_dir or cfg.get("output_dir")

    if data_root is None or not _has_all(data_root):
        data_root = _autodetect_data_root()
    if data_root is None:
        raise FileNotFoundError(
            "Could not locate the Algerian Diac dataset. Set data_root in "
            f"{_DEFAULT_CONFIG}, or pass data_root= explicitly, pointing at a folder containing: "
            f"{REQUIRED_FILES}"
        )

    if output_dir is None:
        output_dir = os.path.join(_REPO_ROOT, "outputs")
    output_dir = os.path.abspath(output_dir)

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    submissions_dir = os.path.join(output_dir, "submissions")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)

    paths = {
        "repo_root": _REPO_ROOT,
        "data_root": data_root,
        "output_dir": output_dir,
        "checkpoints_dir": checkpoints_dir,
        "submissions_dir": submissions_dir,
        "train_jsonl": os.path.join(data_root, "train_Algerian-DIAC.jsonl"),
        "dev_jsonl": os.path.join(data_root, "dev_Algerian-DIAC.jsonl"),
        "vocab_path": os.path.join(data_root, "vocab.json"),
        "labels_path": os.path.join(data_root, "class_labels.txt"),
        "raw_test_txt": os.path.join(data_root, "raw_sentences_test.txt"),
        "raw_test_ids_txt": os.path.join(data_root, "raw_sentences_test_ids.txt"),
        "sample_submission": os.path.join(data_root, "sample_submission.csv"),
        "make_submission_py": os.path.join(data_root, "make_submission.py"),
    }
    return paths


def print_paths(paths):
    print("repo_root       =", paths["repo_root"])
    print("data_root       =", paths["data_root"])
    print("output_dir      =", paths["output_dir"])
    for key in ["train_jsonl", "dev_jsonl", "vocab_path", "labels_path", "raw_test_txt",
                "raw_test_ids_txt", "sample_submission", "make_submission_py"]:
        p = paths[key]
        print(("  [ok]  " if os.path.isfile(p) else "  [MISSING] "), p)
