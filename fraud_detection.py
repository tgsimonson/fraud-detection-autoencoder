"""Credit card fraud detection with a PyOD AutoEncoder.

An autoencoder learns to compress and rebuild transactions. It is trained only
on legitimate transactions (semi-supervised novelty detection), so it learns what
normal spending looks like. Fraudulent transactions differ from that pattern, the
network rebuilds them poorly, and that reconstruction error becomes the anomaly
score.

Dataset:
    Kaggle "Fraud Detection" (whenamancodes/fraud-detection), file creditcard.csv.
    284,807 transactions: Time, V1-V28 (PCA-anonymized features), Amount, and
    Class (1 = fraud, 0 = legitimate).

Usage:
    python fraud_detection.py
    python fraud_detection.py --data path/to/creditcard.csv --epochs 10

Outputs (written to results/):
    metrics.txt                  evaluation summary
    reconstruction_error.png     error distribution for legitimate vs fraud
    precision_recall_curve.png   precision and recall across all thresholds
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files without needing a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyod.models.auto_encoder import AutoEncoder  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# fixed seed so the split and the network weights are reproducible
RANDOM_STATE = 42
TEST_SIZE = 0.3
LABEL_COLUMN = "Class"
# time is seconds since the first transaction, so it describes ordering, not behavior
DROP_COLUMNS = ["Time"]

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "data" / "creditcard.csv"
RESULTS_DIR = PROJECT_DIR / "results"

# chart colors: a colorblind-safe blue and orange pair
COLOR_LEGIT = "#2a78d6"
COLOR_FRAUD = "#eb6834"


def parse_args():
    """Read optional command-line settings."""
    parser = argparse.ArgumentParser(description="Detect credit card fraud with a PyOD AutoEncoder.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="path to creditcard.csv (default: data/creditcard.csv)")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=256, help="mini-batch size (default: 256)")
    return parser.parse_args()


def load_data(path):
    """Load the dataset and split it into features and labels.

    Exits with a clear message if the CSV is missing, since the Kaggle file is
    too large to ship in the repository.
    """
    if not path.exists():
        sys.exit(
            f"Dataset not found at {path}\n"
            "Download creditcard.csv from https://www.kaggle.com/datasets/whenamancodes/fraud-detection\n"
            "and place it in the data/ folder, or pass --data with its location."
        )
    df = pd.read_csv(path)
    features = df.drop(columns=DROP_COLUMNS + [LABEL_COLUMN])
    labels = df[LABEL_COLUMN].to_numpy()
    return features, labels


def split_and_scale(features, labels):
    """Split into stratified train/test sets and standardize the features.

    The scaler is fit on the training set only, so no information from the
    test set leaks into training.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def build_model(contamination, epochs, batch_size):
    """Create the autoencoder.

    The encoder narrows 29 input features down to a 6-unit bottleneck and the
    decoder mirrors it back to 29. The bottleneck forces the network to keep
    only the patterns common to legitimate transactions. Batch normalization and
    dropout are off because the network only needs to reconstruct, not generalize
    across classes, and in testing both blurred the gap between normal and
    fraudulent reconstruction errors.
    """
    return AutoEncoder(
        hidden_neuron_list=[24, 12, 6],
        contamination=contamination,  # expected share of fraud, sets the decision threshold
        epoch_num=epochs,
        batch_size=batch_size,
        dropout_rate=0.0,
        batch_norm=False,
        preprocessing=False,  # features are already standardized above
        random_state=RANDOM_STATE,
        verbose=1,
    )


def evaluate(y_test, scores, predictions):
    """Compute evaluation metrics and return them as printable text.

    ROC AUC and average precision measure how well the scores rank fraud above
    legitimate transactions across every threshold. The classification report
    and confusion matrix show performance at the model's chosen threshold.
    """
    roc_auc = roc_auc_score(y_test, scores)
    avg_precision = average_precision_score(y_test, scores)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    report = classification_report(y_test, predictions, target_names=["Legitimate", "Fraud"], digits=4)
    lines = [
        "PyOD AutoEncoder: credit card fraud detection",
        f"Test transactions: {len(y_test):,} ({int(y_test.sum())} fraud, {y_test.mean():.4%})",
        "",
        f"ROC AUC:            {roc_auc:.4f}",
        f"Average precision:  {avg_precision:.4f}",
        "",
        "Confusion matrix at the model threshold",
        f"  True negatives  (legitimate, passed):   {tn:,}",
        f"  False positives (legitimate, flagged):  {fp:,}",
        f"  False negatives (fraud, missed):        {fn:,}",
        f"  True positives  (fraud, caught):        {tp:,}",
        "",
        "Classification report",
        report,
    ]
    return "\n".join(lines)


def plot_error_distribution(y_test, scores, threshold, path):
    """Plot reconstruction error for legitimate vs fraud transactions."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(np.log10(scores.min()), np.log10(scores.max()), 80)
    ax.hist(scores[y_test == 0], bins=bins, color=COLOR_LEGIT, alpha=0.75, label="Legitimate", density=True)
    ax.hist(scores[y_test == 1], bins=bins, color=COLOR_FRAUD, alpha=0.75, label="Fraud", density=True)
    ax.axvline(threshold, color="#52514e", linestyle="--", linewidth=1.2, label="Model threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction error (anomaly score, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction error by transaction class (test set)", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_precision_recall(y_test, scores, path):
    """Plot the precision-recall curve across all score thresholds."""
    precision, recall, _ = precision_recall_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color=COLOR_LEGIT, linewidth=2)
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of flags that are fraud)")
    ax.set_title(f"Precision-recall curve (AP = {average_precision_score(y_test, scores):.3f})", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    print(f"Loading {args.data} ...")
    features, labels = load_data(args.data)
    print(f"Loaded {len(labels):,} transactions, {int(labels.sum())} fraud ({labels.mean():.4%})")

    x_train, x_test, y_train, y_test = split_and_scale(features, labels)

    # train on legitimate transactions only, so fraud never shapes what "normal" looks like
    x_train_legit = x_train[y_train == 0]

    # the fraud rate sets the threshold: the top 0.17% of normal errors are flagged
    contamination = y_train.mean()
    model = build_model(contamination, args.epochs, args.batch_size)
    print(f"\nTraining AutoEncoder on {len(x_train_legit):,} legitimate transactions "
          f"for {args.epochs} epochs ...")
    model.fit(x_train_legit)

    # reconstruction error for each test transaction, and the 0/1 decision at the threshold
    scores = model.decision_function(x_test)
    predictions = model.predict(x_test)

    summary = evaluate(y_test, scores, predictions)
    print("\n" + summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.txt").write_text(summary + "\n", encoding="utf-8")
    plot_error_distribution(y_test, scores, model.threshold_, RESULTS_DIR / "reconstruction_error.png")
    plot_precision_recall(y_test, scores, RESULTS_DIR / "precision_recall_curve.png")
    print("Saved results/metrics.txt, results/reconstruction_error.png, results/precision_recall_curve.png")


if __name__ == "__main__":
    main()
