"""Evaluation metrics and threshold optimization utilities."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    recall_score,
)

MetricName = Literal["f1", "f1.5", "f2", "recall_constrained"]


def print_section_header(title: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_target_results(
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: Optional[float] = None,
    is_binary: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute and print standardized metrics for a single target."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    if is_binary:
        rec = recall_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        print(f"\n--- {target} ---")
        if threshold is not None:
            print(f"Limiar: {threshold:.4f}")
        print(f"Recall: {rec:.4f} | Acurácia: {acc:.4f}")
        print("Matriz de Confusão:")
        print(cm)
        result = {"target": target, "recall": rec, "accuracy": acc, "confusion_matrix": cm.tolist()}
    else:
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        print(f"\n--- {target} (multiclasse) ---")
        print(f"Acurácia: {acc:.4f} | Weighted Recall: {rec:.4f}")
        print("Matriz de Confusão:")
        print(cm)
        result = {"target": target, "accuracy": acc, "weighted_recall": rec, "confusion_matrix": cm.tolist()}

    if extra:
        for key, value in extra.items():
            print(f"{key}: {value}")
        result.update(extra)

    return result


def _score_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: MetricName,
    beta: float = 2.0,
) -> float:
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "f1.5":
        return float(fbeta_score(y_true, y_pred, beta=1.5, zero_division=0))
    if metric == "f2":
        return float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    return 0.0


def optimize_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: np.ndarray,
    *,
    metric: MetricName = "f1",
    min_recall: float = 0.70,
    beta: float = 2.0,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Sweep probability thresholds and return the optimal value.

    For ``recall_constrained`` (XGBoost-style):
      - Primary: max accuracy with recall >= min_recall
      - Fallback: max F-beta score

    For ``f1``, ``f1.5``, ``f2``: maximize the chosen metric on validation.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs)

    best_thresh = 0.50
    best_score = 0.0
    best_acc_at_recall = 0.0
    fallback_thresh = 0.50
    best_f_beta = 0.0

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        f_beta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

        if f_beta > best_f_beta:
            best_f_beta = f_beta
            fallback_thresh = float(thresh)

        if metric == "recall_constrained":
            if rec >= min_recall and acc > best_acc_at_recall:
                best_acc_at_recall = acc
                best_thresh = float(thresh)
        else:
            score = _score_at_threshold(y_true, y_pred, metric, beta=beta)
            if score > best_score:
                best_score = score
                best_thresh = float(thresh)

    if metric == "recall_constrained":
        if best_acc_at_recall == 0.0:
            final_thresh = fallback_thresh
            best_score = best_f_beta
        else:
            final_thresh = best_thresh
            best_score = best_acc_at_recall
    else:
        final_thresh = best_thresh if best_score > 0 else 0.50

    diagnostics = {
        "best_score": best_score,
        "fallback_threshold": fallback_thresh,
        "best_f_beta": best_f_beta,
    }
    return final_thresh, best_score, diagnostics


def evaluate_with_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Apply threshold and return full metric dict including log loss."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_probs) >= threshold).astype(int)

    return {
        "threshold": threshold,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "log_loss": log_loss(y_true, np.clip(y_probs, 1e-15, 1 - 1e-15)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
