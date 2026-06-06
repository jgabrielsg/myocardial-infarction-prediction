"""TabPFN cloud runner guided by causal DAG feature selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from dotenv import load_dotenv

import tabpfn_client
from tabpfn_client import TabPFNClassifier

from src.causal.dag_generator import CausalDiscoverer
from src.data.data_loader import DataBundle
from src.evaluation.metrics import optimize_threshold, print_section_header, print_target_results
from src.tuning.hparams_store import load_best_hyperparams


class TabPFNCausalRunner:
    """TabPFN with DAG-informed feature filtering and F2/F1.5 threshold optimization."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        load_dotenv()
        self.config_path = Path(config_path)
        with open(self.config_path, encoding="utf-8") as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh)

        token = os.getenv("TABPFN_TOKEN")
        if not token:
            raise EnvironmentError(
                "TABPFN_TOKEN not set. Copy .env.example to .env and add your token."
            )
        tabpfn_client.set_access_token(token)

        thresh_cfg = self.config["thresholds"]["tabpfn_causal"]
        hp = load_best_hyperparams().get("tabpfn_causal", {})
        if hp:
            self.thresholds = np.arange(hp["thresh_start"], hp["thresh_stop"], hp["thresh_step"])
            self.metric = hp.get("threshold_metric", "f2")
            self.top_k_parents = int(hp.get("top_k_parents", self.config["causal"].get("top_k_parents", 10)))
            print(f"Using tuned TabPFN causal params: metric={self.metric}, top_k={self.top_k_parents}")
        else:
            self.thresholds = np.arange(thresh_cfg["start"], thresh_cfg["stop"], thresh_cfg["step"])
            self.metric = thresh_cfg.get("metric", "f2")
            self.top_k_parents = self.config["causal"].get("top_k_parents", 10)

    def run(self, bundle: DataBundle) -> List[Dict[str, Any]]:
        """Train DAG-filtered TabPFN and evaluate all targets."""
        print_section_header("TABPFN CAUSAL — DAG-Informed Features")

        causal_parents = CausalDiscoverer.load_parents(bundle, self.config_path)
        results: List[Dict[str, Any]] = []

        for target in bundle.target_names:
            causal_features = causal_parents.get(target, [])[: self.top_k_parents]
            print(f"\n--- Processing target: {target} ---")
            print(f"Causal features: {len(causal_features)}")

            if causal_features:
                X_train = bundle.X_train_base[causal_features]
                X_val = bundle.X_val_base[causal_features]
                X_test = bundle.X_test_base[causal_features]
            else:
                print("  Warning: no causal parents found — falling back to all adm_cols.")
                X_train = bundle.X_train_base[bundle.adm_cols]
                X_val = bundle.X_val_base[bundle.adm_cols]
                X_test = bundle.X_test_base[bundle.adm_cols]

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
                        "Causal Features": len(causal_features),
                        "Mode": "CAUSAL",
                    },
                )
                results.append(result)
            else:
                preds = model.predict(X_test)
                y_test = bundle.y_test_base[target].astype(int).values
                result = print_target_results(
                    target,
                    y_test,
                    preds,
                    is_binary=False,
                    extra={"Causal Features": len(causal_features), "Mode": "CAUSAL"},
                )
                results.append(result)

        return results
