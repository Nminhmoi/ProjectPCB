"""Huấn luyện + đánh giá phân loại khuyết tật (SVM RBF vs Random Forest, cv2.ml).

Chia train/test theo `pair_id` (GroupKFold) để ảnh cùng bo mạch không nằm cả hai
phía. Báo cáo accuracy, F1 macro, precision/recall từng lớp, confusion matrix;
lưu model tốt nhất vào `model_classifier.pkl` và biểu đồ vào `experiments/charts/`.

    python experiments/exp_classify.py --limit-rows 300      # thử code
    python experiments/exp_classify.py                       # full dataset
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "MODULE")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from MODULE.module_classification import CLASS_NAMES, FEATURE_NAMES, save_model

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "classifier_dataset.csv"
DEFAULT_MODEL = SCRIPT_DIR.parent / "model_classifier.pkl"
DEFAULT_CHARTS = SCRIPT_DIR / "charts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--charts-dir", type=Path, default=DEFAULT_CHARTS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=None,
                        help="Chỉ dùng N dòng đầu (thử code).")
    parser.add_argument("--drop-background", action="store_true",
                        help="Bỏ lớp background (chỉ phân loại 6 lớp lỗi).")
    parser.add_argument("--results-csv", type=Path,
                        default=SCRIPT_DIR / "classify_results.csv")
    parser.add_argument("--rf-trees", type=int, default=60,
                        help="Số cây tối đa của Random Forest (model .xml càng nhỏ càng nhanh).")
    parser.add_argument("--balance", action="store_true",
                        help="Oversample lớp thiểu số khi train (mặc định: tắt).")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------
def load_dataset(path: Path, limit_rows=None, drop_background=False):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if drop_background:
        rows = [r for r in rows if r["label"] != "background"]
    if limit_rows:
        rows = rows[:limit_rows]
    X = np.array([[float(r[name]) for name in FEATURE_NAMES] for r in rows], np.float32)
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["pair_id"] for r in rows])
    return X, y, groups


def group_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Chia fold theo pair_id, cân bằng số dòng mỗi fold."""
    unique = sorted(set(groups.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    sizes = [0] * k
    fold_of = {}
    for group in unique:
        target = int(np.argmin(sizes))
        fold_of[group] = target
        sizes[target] += int(np.count_nonzero(groups == group))
    return np.array([fold_of[g] for g in groups.tolist()])


def standardize(train: np.ndarray, other: np.ndarray):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (other - mean) / std, {"mean": mean, "std": std}


def oversample(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Lặp mẫu lớp thiểu số cho cân bằng (chỉ trên tập train của mỗi fold)."""
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    parts_x, parts_y = [], []
    for name, count in zip(classes, counts):
        index = np.flatnonzero(y == name)
        parts_x.append(X[index])
        parts_y.append(np.full(len(index), name, dtype=y.dtype))
        if count < target:
            extra = rng.choice(index, target - count, replace=True)
            parts_x.append(X[extra])
            parts_y.append(np.full(len(extra), name, dtype=y.dtype))
    return np.vstack(parts_x), np.concatenate(parts_y)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def train_svm(X, y, c_value: float, gamma: float):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(c_value)
    svm.setGamma(gamma)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 3000, 1e-4))
    svm.train(X, cv2.ml.ROW_SAMPLE, np.asarray(y, np.int32).reshape(-1, 1))
    return svm


def train_rtrees(X, y, max_depth: int, trees: int = 120):
    forest = cv2.ml.RTrees_create()
    forest.setMaxDepth(max_depth)
    forest.setMinSampleCount(2)
    forest.setCVFolds(0)
    forest.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, trees, 0.01))
    forest.train(X, cv2.ml.ROW_SAMPLE, np.asarray(y, np.int32).reshape(-1, 1))
    return forest


def predict_index(model, X) -> np.ndarray:
    output = model.predict(np.asarray(X, np.float32))
    values = output[1] if isinstance(output, tuple) else output
    return np.rint(np.asarray(values).ravel()).astype(int)


# ---------------------------------------------------------------------
# Chỉ số
# ---------------------------------------------------------------------
def confusion_matrix(y_true_idx, y_pred_idx, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for truth, pred in zip(y_true_idx, y_pred_idx):
        if 0 <= truth < n_classes and 0 <= pred < n_classes:
            matrix[truth, pred] += 1
    return matrix


def per_class_scores(matrix: np.ndarray, names):
    rows = []
    for i, name in enumerate(names):
        tp = matrix[i, i]
        support = matrix[i, :].sum()
        predicted = matrix[:, i].sum()
        precision = tp / predicted if predicted else float("nan")
        recall = tp / support if support else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if support and predicted and precision + recall > 0 else 0.0)
        rows.append({"class": name, "support": int(support), "precision": precision,
                     "recall": recall, "f1": f1})
    return rows


def macro_f1(rows) -> float:
    values = [r["f1"] for r in rows if r["support"] > 0]
    return float(np.mean(values)) if values else 0.0


def print_scores(rows, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'lop':<16} {'support':>8} {'precision':>10} {'recall':>8} {'f1':>8}")
    for row in rows:
        print(f"  {row['class']:<16} {row['support']:>8} {row['precision']:>10.3f} "
              f"{row['recall']:>8.3f} {row['f1']:>8.3f}")
    print(f"  {'macro F1':<16} {'':>8} {'':>10} {'':>8} {macro_f1(rows):>8.3f}")


def save_confusion_png(matrix, names, path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[CANH BAO] Khong co matplotlib - bo qua ve confusion matrix.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix: {path}")


# ---------------------------------------------------------------------
# Grid + CV
# ---------------------------------------------------------------------
def build_configs(X, trees: int = 120) -> list:
    gamma_auto = 1.0 / (X.shape[1] * float(np.var(X)) or 1.0)
    configs = []
    for c_value in (1.0, 10.0, 100.0):
        configs.append({"name": f"SVM_C{c_value:g}", "kind": "svm",
                        "c": c_value, "gamma": gamma_auto})
    for depth in (12, 20):
        configs.append({"name": f"RF_depth{depth}", "kind": "rf",
                        "depth": depth, "trees": trees})
    return configs


def fit_model(config, X, y):
    if config["kind"] == "svm":
        return train_svm(X, y, config["c"], config["gamma"])
    return train_rtrees(X, y, config["depth"], config.get("trees", 120))


def run_fold(config, X_train, y_train_idx, X_test, rng, balance):
    X_train_s, X_test_s, _ = standardize(X_train, X_test)
    if balance:
        X_train_s, y_train_idx = oversample(X_train_s, y_train_idx, rng)
    model = fit_model(config, X_train_s, y_train_idx)
    return predict_index(model, X_test_s)


def main() -> int:
    args = parse_args()
    if not Path(args.dataset).is_file():
        print(f"[LOI] Khong thay dataset: {args.dataset}. "
              f"Chay experiments/build_classifier_dataset.py truoc.")
        return 1

    X, y, groups = load_dataset(args.dataset, args.limit_rows, args.drop_background)
    names = [n for n in CLASS_NAMES if n in set(y.tolist())]
    index_of = {name: i for i, name in enumerate(names)}
    y_idx = np.array([index_of[v] for v in y])
    folds = group_folds(groups, args.folds, args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"Dataset      : {args.dataset}")
    print(f"Dong / cap   : {len(y)} / {len(set(groups.tolist()))} | folds = {args.folds}")
    print("Phan bo      :", {n: int((y == n).sum()) for n in names})
    print(f"Dac trung    : {X.shape[1]} | lop: {len(names)}")

    configs = build_configs(X, args.rf_trees)
    results = []
    start = time.time()

    for config in configs:
        oof_pred = np.full(len(y), -1, dtype=int)
        fold_scores = []
        for fold in range(args.folds):
            train_index = np.flatnonzero(folds != fold)
            test_index = np.flatnonzero(folds == fold)
            if not len(test_index):
                continue
            pred = run_fold(config, X[train_index], y_idx[train_index],
                            X[test_index], rng, args.balance)
            oof_pred[test_index] = pred
            matrix = confusion_matrix(y_idx[test_index], pred, len(names))
            fold_scores.append(macro_f1(per_class_scores(matrix, names)))

        valid = oof_pred >= 0
        matrix = confusion_matrix(y_idx[valid], oof_pred[valid], len(names))
        rows = per_class_scores(matrix, names)
        accuracy = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0
        score = macro_f1(rows)
        results.append({"config": config, "macro_f1": score, "accuracy": accuracy,
                        "matrix": matrix, "rows": rows,
                        "fold_mean": float(np.mean(fold_scores)),
                        "fold_std": float(np.std(fold_scores))})
        print(f"[{config['name']:<12}] macro F1 = {score:.3f} "
              f"(fold {np.mean(fold_scores):.3f} +/- {np.std(fold_scores):.3f}), "
              f"acc = {accuracy:.3f}")

    best = max(results, key=lambda item: item["macro_f1"])
    print(f"\n>>> Chon {best['config']['name']} (macro F1 {best['macro_f1']:.3f})")
    print_scores(best["rows"], "Chi so theo lop (out-of-fold):")
    print("\nConfusion matrix (hang = that, cot = du doan):")
    print("  " + "".join(f"{name[:9]:>11}" for name in names))
    for name, line in zip(names, best["matrix"]):
        print(f"  {name[:14]:<14}" + "".join(f"{v:>11d}" for v in line))

    save_confusion_png(best["matrix"], names, Path(args.charts_dir) / "confusion_matrix.png",
                       f"{best['config']['name']} - macro F1 {best['macro_f1']:.3f}")

    # Model cuoi: train tren toan bo du lieu, luu kem scaler de classify dung lai.
    X_all, _, scaler = standardize(X, X)
    if args.balance:
        X_all, y_train_all = oversample(X_all, y_idx, rng)
    else:
        y_train_all = y_idx
    model = fit_model(best["config"], X_all, y_train_all)
    path = save_model({
        "model": model,
        "scaler": scaler,
        "classes": names,
        "feature_names": FEATURE_NAMES,
        "meta": {
            "config": dict(best["config"]),
            "folds": args.folds,
            "macro_f1": best["macro_f1"],
            "accuracy": best["accuracy"],
            "n_rows": int(len(y)),
            "n_pairs": int(len(set(groups.tolist()))),
        },
    }, args.model_out)
    print(f"Model: {path}")

    results_csv = Path(args.results_csv)
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "macro_f1", "accuracy", "fold_mean", "fold_std"]
                        + [f"{r['class']}_f1" for r in best["rows"]])
        for item in results:
            writer.writerow([item["config"]["name"], f"{item['macro_f1']:.4f}",
                             f"{item['accuracy']:.4f}", f"{item['fold_mean']:.4f}",
                             f"{item['fold_std']:.4f}"]
                            + [f"{r['f1']:.4f}" for r in item["rows"]])
    print(f"Bang so sanh: {results_csv}")
    print(f"Tong thoi gian: {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
