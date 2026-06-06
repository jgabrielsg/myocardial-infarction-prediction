"""TabPFN cloud runner using all admission features (free mode)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from dotenv import load_dotenv
import tabpfn_client
from tabpfn_client import TabPFNClassifier

from src.data.data_loader import DataBundle
from src.evaluation.metrics import optimize_threshold, print_section_header, print_target_results
from src.tuning.hparams_store import load_best_hyperparams


class TabPFNFreeRunner:
    """TabPFN inference on all admission columns with F1 threshold optimization."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        load_dotenv()
        with open(config_path, encoding="utf-8") as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh)

        token = os.getenv("TABPFN_TOKEN")
        if not token:
            raise EnvironmentError(
                "TABPFN_TOKEN not set. Copy .env.example to .env and add your token."
            )
        tabpfn_client.set_access_token(token)

        if os.getenv("TABPFN_ALLOW_CPU_LARGE_DATASET"):
            os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = os.getenv(
                "TABPFN_ALLOW_CPU_LARGE_DATASET", "1"
            )

        thresh_cfg = self.config["thresholds"]["tabpfn_free"]
        hp = load_best_hyperparams().get("tabpfn_free", {})
        if hp:
            self.thresholds = np.arange(hp["thresh_start"], hp["thresh_stop"], hp["thresh_step"])
            self.metric = hp.get("threshold_metric", "f1")
            print(f"Using tuned TabPFN free params: metric={self.metric}, range=[{hp['thresh_start']}, {hp['thresh_stop']})")
        else:
            self.thresholds = np.arange(thresh_cfg["start"], thresh_cfg["stop"], thresh_cfg["step"])
            self.metric = thresh_cfg.get("metric", "f1")

    def run(self, bundle: DataBundle) -> List[Dict[str, Any]]:
        """Train TabPFN on admission features and evaluate all targets."""
        print_section_header("TABPFN FREE — All Admission Features")

        X_train = bundle.X_train_base[bundle.adm_cols].copy()
        X_val = bundle.X_val_base[bundle.adm_cols].copy()
        X_test = bundle.X_test_base[bundle.adm_cols].copy()

        print(f"Admission features used: {len(bundle.adm_cols)}")

        results: List[Dict[str, Any]] = []

        for target in bundle.target_names:
            print(f"\n--- Processing target: {target} ---")

            model = TabPFNClassifier()
            model.fit(X_train, bundle.y_train_base[target].astype(int))

            if target in bundle.binary_targets:
                val_probs = model.predict_proba(X_val)[:, 1]
                y_val = bundle.y_val_base[target].astype(int).values

                final_thresh, best_score, _ = optimize_threshold(
                    y_val,
                    val_probs,
                    self.thresholds,
                    metric=self.metric,
                )

                test_probs = model.predict_proba(X_test)[:, 1]
                y_test = bundle.y_test_base[target].astype(int).values
                y_pred = (test_probs >= final_thresh).astype(int)

                result = print_target_results(
                    target,
                    y_test,
                    y_pred,
                    threshold=final_thresh,
                    extra={
                        f"{self.metric.upper()}-Score (Val)": f"{best_score:.4f}",
                        "Mode": "FREE",
                    },
                )
                results.append(result)
            else:
                preds = model.predict(X_test)
                y_test = bundle.y_test_base[target].astype(int).values
                result = print_target_results(target, y_test, preds, is_binary=False, extra={"Mode": "FREE"})
                results.append(result)

        return results
