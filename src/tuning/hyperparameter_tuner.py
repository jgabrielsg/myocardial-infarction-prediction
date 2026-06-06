"""Hyperparameter tuning for all five pipeline models using sequential Optuna."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import yaml
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

from src.causal.dag_generator import CausalDiscoverer
from src.data.data_loader import DataBundle, DataLoader
from src.evaluation.metrics import optimize_threshold
from src.models.pytorch_net import MultiTaskMI
from src.tuning.hparams_store import save_best_hyperparams
from src.tuning.sequential_optuna import SequentialOptunaTuner
from src.utils.device import device_summary, get_device, get_xgboost_device, setup_cuda

try:
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier
except ImportError:
    tabpfn_client = None
    TabPFNClassifier = None


class HyperparameterTuner:
    """Run sequential Optuna tuning for each model; evaluate on held-out test set."""

    XGB_DEFAULTS = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 1e-8,
        "reg_lambda": 1.0,
    }
    PYTORCH_DEFAULTS = {
        "learning_rate": 0.001,
        "dropout": 0.3,
        "batch_size": 64,
        "weight_decay": 1e-4,
    }
    AG_DEFAULTS = {"presets": "best_quality", "time_limit": 60}
    TABPFN_FREE_DEFAULTS = {
        "threshold_metric": "f1",
        "thresh_start": 0.001,
        "thresh_stop": 0.95,
        "thresh_step": 0.001,
    }
    TABPFN_CAUSAL_DEFAULTS = {
        "top_k_parents": 10,
        "threshold_metric": "f2",
        "thresh_start": 0.0001,
        "thresh_stop": 0.5,
        "thresh_step": 0.0001,
    }

    @staticmethod
    def _merge_params(params: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        merged = defaults.copy()
        merged.update({k: v for k, v in params.items() if not k.startswith("_")})
        return merged

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        with open(config_path, encoding="utf-8") as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh)

        self.config_path = Path(config_path)
        self.random_state = self.config["data"]["random_state"]
        tune_cfg = self.config.get("tuning", {})
        self.n_trials_per_param = tune_cfg.get("n_trials_per_param", 15)
        self.output_path = Path(tune_cfg.get("output_path", "data/processed/best_hyperparams.json"))
        self.results_path = Path(tune_cfg.get("results_path", "results/tuning_results.json"))

        setup_cuda()
        print(device_summary())

    def run_all(self, bundle: DataBundle) -> Dict[str, Any]:
        all_results: Dict[str, Any] = {}

        all_results["xgboost"] = self.tune_xgboost(bundle)
        all_results["pytorch"] = self.tune_pytorch(bundle)
        all_results["autogluon"] = self.tune_autogluon(bundle)
        all_results["tabpfn_free"] = self.tune_tabpfn_free(bundle)
        all_results["tabpfn_causal"] = self.tune_tabpfn_causal(bundle)

        save_best_hyperparams(all_results, self.output_path)
        self._save_test_results(all_results)
        self._print_summary(all_results)
        return all_results

    def _admission_frames(self, bundle: DataBundle) -> Dict[str, pd.DataFrame]:
        return {
            "train": bundle.get_admission_features(bundle.X_train_base).apply(pd.to_numeric, errors="coerce"),
            "val": bundle.get_admission_features(bundle.X_tune_val).apply(pd.to_numeric, errors="coerce"),
            "test": bundle.get_admission_features(bundle.X_test_base).apply(pd.to_numeric, errors="coerce"),
            "tune_train": bundle.get_admission_features(bundle.X_tune_train).apply(pd.to_numeric, errors="coerce"),
        }

    def _macro_f2_binary(
        self,
        bundle: DataBundle,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        predict_fn: Callable[[str], np.ndarray],
    ) -> float:
        scores = []
        for target in bundle.binary_targets:
            y_true = y_val[target].astype(int).values
            y_pred = predict_fn(target)
            scores.append(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
        return float(np.mean(scores)) if scores else 0.0

    def _eval_on_test(
        self,
        bundle: DataBundle,
        model_name: str,
        scores: Dict[str, float],
    ) -> Dict[str, Any]:
        return {"model": model_name, "test_scores": scores}

    # ------------------------------------------------------------------ XGBoost
    def tune_xgboost(self, bundle: DataBundle) -> Dict[str, Any]:
        print("\n" + "=" * 70)
        print("TUNING: XGBoost (sequential Optuna)")
        print("=" * 70)

        frames = self._admission_frames(bundle)
        tree_method, device = get_xgboost_device()
        xgb_cfg = self.config["xgboost"]
        min_recall = self.config["thresholds"]["min_recall_acceptable"]
        thresh_cfg = self.config["thresholds"]["xgboost"]
        thresholds = np.arange(thresh_cfg["start"], thresh_cfg["stop"], thresh_cfg["step"])

        probe_targets = bundle.binary_targets[:3]

        def objective(params: Dict[str, Any]) -> float:
            p = self._merge_params(params, self.XGB_DEFAULTS)
            fold_scores = []
            for target in probe_targets:
                y_tr = bundle.y_tune_train[target].astype(int)
                y_va = bundle.y_tune_val[target].astype(int)
                weights = compute_sample_weight("balanced", y_tr)

                model = xgb.XGBClassifier(
                    n_estimators=xgb_cfg["n_estimators_search"],
                    max_depth=int(p["max_depth"]),
                    learning_rate=p["learning_rate"],
                    subsample=p["subsample"],
                    colsample_bytree=p["colsample_bytree"],
                    min_child_weight=int(p["min_child_weight"]),
                    reg_alpha=p["reg_alpha"],
                    reg_lambda=p["reg_lambda"],
                    tree_method=tree_method,
                    device=device,
                    random_state=self.random_state,
                    n_jobs=-1,
                    eval_metric="aucpr",
                    early_stopping_rounds=50,
                )
                model.fit(
                    frames["tune_train"],
                    y_tr,
                    sample_weight=weights,
                    eval_set=[(frames["val"], y_va)],
                    verbose=False,
                )
                val_probs = model.predict_proba(frames["val"])[:, 1]
                thresh, _, _ = optimize_threshold(
                    y_va.values, val_probs, thresholds,
                    metric="recall_constrained", min_recall=min_recall, beta=2.0,
                )
                y_pred = (val_probs >= thresh).astype(int)
                fold_scores.append(fbeta_score(y_va, y_pred, beta=2, zero_division=0))
            return float(np.mean(fold_scores))

        def suggest(trial: optuna.Trial, name: str) -> Any:
            specs = {
                "max_depth": lambda: trial.suggest_int("max_depth", 3, 9),
                "learning_rate": lambda: trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "subsample": lambda: trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": lambda: trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": lambda: trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": lambda: trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": lambda: trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            return specs[name]()

        param_order = [
            "learning_rate", "max_depth", "min_child_weight",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        ]
        best = SequentialOptunaTuner(
            param_order, suggest, objective,
            n_trials_per_param=self.n_trials_per_param,
            seed=self.random_state,
        ).optimize()

        test_score = self._xgboost_test_score(bundle, frames, best, tree_method, device, thresholds, min_recall)
        best["val_macro_f2"] = best.pop("_best_score", 0)
        best["test_macro_f2"] = test_score
        best.pop("_tuning_history", None)
        return best

    def _xgboost_test_score(
        self, bundle, frames, params, tree_method, device, thresholds, min_recall,
    ) -> float:
        params = self._merge_params(params, self.XGB_DEFAULTS)
        scores = []
        for target in bundle.binary_targets:
            y_tr = bundle.y_train_base[target].astype(int)
            y_te = bundle.y_test_base[target].astype(int)
            y_va = bundle.y_tune_val[target].astype(int)
            weights = compute_sample_weight("balanced", y_tr)

            model = xgb.XGBClassifier(
                n_estimators=self.config["xgboost"]["n_estimators_tune"],
                max_depth=int(params["max_depth"]),
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                min_child_weight=int(params["min_child_weight"]),
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                tree_method=tree_method,
                device=device,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric="aucpr",
                early_stopping_rounds=100,
            )
            model.fit(
                frames["train"], y_tr, sample_weight=weights,
                eval_set=[(frames["val"], y_va)], verbose=False,
            )
            n_trees = max(model.best_iteration, 1)
            final = xgb.XGBClassifier(
                n_estimators=n_trees, max_depth=int(params["max_depth"]),
                learning_rate=params["learning_rate"], subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                min_child_weight=int(params["min_child_weight"]),
                reg_alpha=params["reg_alpha"], reg_lambda=params["reg_lambda"],
                tree_method=tree_method, device=device, random_state=self.random_state, n_jobs=-1,
            )
            final.fit(frames["train"], y_tr, sample_weight=weights)
            val_probs = final.predict_proba(frames["val"])[:, 1]
            test_probs = final.predict_proba(frames["test"])[:, 1]
            thresh, _, _ = optimize_threshold(
                y_va.values, val_probs, thresholds,
                metric="recall_constrained", min_recall=min_recall, beta=2.0,
            )
            y_pred = (test_probs >= thresh).astype(int)
            scores.append(fbeta_score(y_te, y_pred, beta=2, zero_division=0))
        return float(np.mean(scores))

    # ------------------------------------------------------------------ PyTorch
    def tune_pytorch(self, bundle: DataBundle) -> Dict[str, Any]:
        print("\n" + "=" * 70)
        print("TUNING: PyTorch Multi-Task NN (sequential Optuna + CUDA)")
        print("=" * 70)

        device = get_device()
        pt_cfg = self.config["pytorch"]
        loader = DataLoader()

        X_aug, y_aug = loader.create_augmented_dataset(bundle.X_train_base, bundle.y_train_base, bundle)
        cat_cols = bundle.categorical_cols + ["TIMELINE_STAGE"]
        num_cols = bundle.numeric_cols

        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ])
        X_proc = preprocessor.fit_transform(X_aug)
        y_bin = y_aug[bundle.binary_targets].values.astype(np.float32)
        y_multi = y_aug[bundle.multiclass_target].values.astype(np.int64)

        X_tr, X_va, y_b_tr, y_b_va, y_m_tr, y_m_va = train_test_split(
            X_proc, y_bin, y_multi, test_size=0.2, random_state=self.random_state,
        )

        def objective(params: Dict[str, Any]) -> float:
            p = self._merge_params(params, self.PYTORCH_DEFAULTS)
            bs = int(p["batch_size"])
            X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
            y_bt = torch.tensor(y_b_tr, dtype=torch.float32, device=device)
            y_mt = torch.tensor(y_m_tr, dtype=torch.long, device=device)
            X_v = torch.tensor(X_va, dtype=torch.float32, device=device)
            y_bv = torch.tensor(y_b_va, dtype=torch.float32, device=device)
            y_mv = torch.tensor(y_m_va, dtype=torch.long, device=device)

            train_ld = TorchDataLoader(TensorDataset(X_t, y_bt, y_mt), batch_size=bs, shuffle=True)
            val_ld = TorchDataLoader(TensorDataset(X_v, y_bv, y_mv), batch_size=bs, shuffle=False)

            model = MultiTaskMI(
                X_tr.shape[1], len(bundle.binary_targets),
                pt_cfg["num_multiclass_classes"], dropout_rate=p["dropout"],
            ).to(device)

            pos_w = (y_b_tr.shape[0] - y_b_tr.sum(axis=0)) / (y_b_tr.sum(axis=0) + 1e-5)
            pos_w = torch.tensor(pos_w * pt_cfg["pos_weight_multiplier"], device=device)
            crit_bin = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            crit_multi = nn.CrossEntropyLoss()
            opt = optim.Adam(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])

            for _ in range(pt_cfg["tune_epochs"]):
                model.train()
                for bx, ybb, ymm in train_ld:
                    opt.zero_grad()
                    ob, om = model(bx)
                    loss = crit_bin(ob, ybb) + crit_multi(om, ymm) * pt_cfg["multiclass_loss_weight"]
                    loss.backward()
                    opt.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, ybb, ymm in val_ld:
                    ob, om = model(bx)
                    loss = crit_bin(ob, ybb) + crit_multi(om, ymm) * pt_cfg["multiclass_loss_weight"]
                    val_loss += loss.item()
            return -val_loss / max(len(val_ld), 1)

        def suggest(trial: optuna.Trial, name: str) -> Any:
            specs = {
                "learning_rate": lambda: trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "dropout": lambda: trial.suggest_float("dropout", 0.1, 0.6),
                "batch_size": lambda: trial.suggest_categorical("batch_size", [32, 64, 128]),
                "weight_decay": lambda: trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            }
            return specs[name]()

        best = SequentialOptunaTuner(
            ["learning_rate", "dropout", "batch_size", "weight_decay"],
            suggest, objective,
            n_trials_per_param=self.n_trials_per_param,
            seed=self.random_state,
        ).optimize()

        test_f2 = self._pytorch_test_f2(bundle, preprocessor, best, device)
        best["val_neg_loss"] = best.pop("_best_score", 0)
        best["test_macro_f2"] = test_f2
        best.pop("_tuning_history", None)
        return best

    def _pytorch_test_f2(self, bundle, preprocessor, params, device) -> float:
        loader = DataLoader()
        pt_cfg = self.config["pytorch"]
        X_aug, y_aug = loader.create_augmented_dataset(bundle.X_train_base, bundle.y_train_base, bundle)
        X_proc = preprocessor.fit_transform(X_aug)
        y_bin = y_aug[bundle.binary_targets].values.astype(np.float32)
        y_multi = y_aug[bundle.multiclass_target].values.astype(np.int64)

        X_t = torch.tensor(X_proc, dtype=torch.float32, device=device)
        y_bt = torch.tensor(y_bin, dtype=torch.float32, device=device)
        y_mt = torch.tensor(y_multi, dtype=torch.long, device=device)
        train_ld = TorchDataLoader(
            TensorDataset(X_t, y_bt, y_mt), batch_size=int(params["batch_size"]), shuffle=True,
        )

        model = MultiTaskMI(
            X_proc.shape[1], len(bundle.binary_targets),
            pt_cfg["num_multiclass_classes"], dropout_rate=params["dropout"],
        ).to(device)
        pos_w = (y_bin.shape[0] - y_bin.sum(axis=0)) / (y_bin.sum(axis=0) + 1e-5)
        pos_w = torch.tensor(pos_w * pt_cfg["pos_weight_multiplier"], device=device)
        crit_bin = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        crit_multi = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

        for _ in range(min(pt_cfg["max_epochs"], 80)):
            model.train()
            for bx, ybb, ymm in train_ld:
                opt.zero_grad()
                ob, om = model(bx)
                loss = crit_bin(ob, ybb) + crit_multi(om, ymm) * pt_cfg["multiclass_loss_weight"]
                loss.backward()
                opt.step()

        X_test_adm = loader.generate_temporal_datasets(bundle.X_test_base, bundle)["admission"].copy()
        X_test_adm["TIMELINE_STAGE"] = 0
        for col in preprocessor.feature_names_in_:
            if col not in X_test_adm.columns:
                X_test_adm[col] = np.nan
        X_test = torch.tensor(preprocessor.transform(X_test_adm[preprocessor.feature_names_in_]), dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test)[0]).cpu().numpy()

        comp = self.config["thresholds"]["comparison"]
        thresholds = np.arange(comp["start"], comp["stop"], comp["step"])
        scores = []
        for i, target in enumerate(bundle.binary_targets):
            y_true = bundle.y_test_base[target].values.astype(int)
            thresh, _, _ = optimize_threshold(y_true, probs[:, i], thresholds, metric="recall_constrained", min_recall=comp["min_recall"])
            scores.append(fbeta_score(y_true, (probs[:, i] >= thresh).astype(int), beta=2, zero_division=0))
        return float(np.mean(scores))

    # ------------------------------------------------------------------ AutoGluon
    def tune_autogluon(self, bundle: DataBundle) -> Dict[str, Any]:
        print("\n" + "=" * 70)
        print("TUNING: AutoGluon Classifier Chain (sequential Optuna)")
        print("=" * 70)

        X_adm = bundle.get_admission_features(bundle.X_full)
        df = pd.concat([X_adm, bundle.y_full], axis=1)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=self.random_state)
        tune_dir = Path("AutogluonModels/TuningProbe")
        tune_dir.mkdir(parents=True, exist_ok=True)

        probe_target = bundle.binary_targets[0]

        def objective(params: Dict[str, Any]) -> float:
            p = self._merge_params(params, self.AG_DEFAULTS)
            path = str(tune_dir / f"probe_{p['presets']}_{p['time_limit']}")
            train_input = TabularDataset(train_df.drop(columns=[c for c in bundle.target_names if c != probe_target]))
            if os.path.exists(path):
                import shutil
                shutil.rmtree(path, ignore_errors=True)

            num_pos = (train_input[probe_target] == 1).sum()
            num_neg = (train_input[probe_target] == 0).sum()
            weight_ratio = num_neg / (num_pos + 1e-5)
            train_input["sample_weight"] = np.where(train_input[probe_target] == 1, weight_ratio, 1.0)

            predictor = TabularPredictor(
                label=probe_target, eval_metric="roc_auc", problem_type="binary",
                sample_weight="sample_weight", path=path,
            ).fit(
                train_input,
                presets=p["presets"],
                time_limit=int(p["time_limit"]),
                excluded_model_types=self.config["autogluon"]["excluded_model_types"],
                verbosity=0,
            )
            val_input = val_df.drop(columns=[c for c in bundle.target_names if c != probe_target])
            probs = predictor.predict_proba(val_input).iloc[:, 1].values
            y_true = val_df[probe_target].astype(int).values
            comp = self.config["thresholds"]["comparison"]
            thresholds = np.arange(comp["start"], comp["stop"], comp["step"])
            thresh, _, _ = optimize_threshold(y_true, probs, thresholds, metric="recall_constrained", min_recall=comp["min_recall"])
            return float(fbeta_score(y_true, (probs >= thresh).astype(int), beta=2, zero_division=0))

        def suggest(trial: optuna.Trial, name: str) -> Any:
            if name == "presets":
                return trial.suggest_categorical("presets", ["medium_quality", "good_quality", "best_quality"])
            if name == "time_limit":
                return trial.suggest_int("time_limit", 30, 180, step=30)
            raise KeyError(name)

        best = SequentialOptunaTuner(
            ["presets", "time_limit"], suggest, objective,
            n_trials_per_param=max(5, self.n_trials_per_param // 2),
            seed=self.random_state,
        ).optimize()

        best["val_f2_probe"] = best.pop("_best_score", 0)
        best.pop("_tuning_history", None)
        return best

    # ------------------------------------------------------------------ TabPFN
    def _ensure_tabpfn_token(self) -> None:
        if tabpfn_client is None:
            raise ImportError("tabpfn-client is required for TabPFN tuning.")
        token = os.getenv("TABPFN_TOKEN")
        if not token:
            raise EnvironmentError("TABPFN_TOKEN not set in .env")
        tabpfn_client.set_access_token(token)

    def tune_tabpfn_free(self, bundle: DataBundle) -> Dict[str, Any]:
        print("\n" + "=" * 70)
        print("TUNING: TabPFN Free (threshold + metric)")
        print("=" * 70)
        self._ensure_tabpfn_token()

        X_train = bundle.X_train_base[bundle.adm_cols]
        X_val = bundle.X_val_base[bundle.adm_cols]
        X_test = bundle.X_test_base[bundle.adm_cols]

        def objective(params: Dict[str, Any]) -> float:
            p = self._merge_params(params, self.TABPFN_FREE_DEFAULTS)
            metric = p["threshold_metric"]
            start, stop, step = p["thresh_start"], p["thresh_stop"], p["thresh_step"]
            thresholds = np.arange(start, stop, step)
            scores = []
            for target in bundle.binary_targets[:3]:
                model = TabPFNClassifier()
                model.fit(X_train, bundle.y_train_base[target].astype(int))
                probs = model.predict_proba(X_val)[:, 1]
                y_true = bundle.y_val_base[target].astype(int).values
                best_s, best_t = 0.0, 0.5
                for t in thresholds:
                    pred = (probs >= t).astype(int)
                    s = f1_score(y_true, pred, zero_division=0) if metric == "f1" else fbeta_score(y_true, pred, beta=1.5, zero_division=0)
                    if s > best_s:
                        best_s, best_t = s, t
                scores.append(best_s)
            return float(np.mean(scores))

        def suggest(trial: optuna.Trial, name: str) -> Any:
            specs = {
                "threshold_metric": lambda: trial.suggest_categorical("threshold_metric", ["f1", "f1.5"]),
                "thresh_start": lambda: trial.suggest_float("thresh_start", 0.001, 0.05, log=True),
                "thresh_stop": lambda: trial.suggest_float("thresh_stop", 0.5, 0.95),
                "thresh_step": lambda: trial.suggest_categorical("thresh_step", [0.001, 0.005, 0.01]),
            }
            return specs[name]()

        best = SequentialOptunaTuner(
            ["threshold_metric", "thresh_start", "thresh_stop", "thresh_step"],
            suggest, objective,
            n_trials_per_param=max(5, self.n_trials_per_param // 3),
            seed=self.random_state,
        ).optimize()

        test_f2 = self._tabpfn_test(bundle, best, causal=False)
        best["val_score"] = best.pop("_best_score", 0)
        best["test_macro_f2"] = test_f2
        best.pop("_tuning_history", None)
        return best

    def tune_tabpfn_causal(self, bundle: DataBundle) -> Dict[str, Any]:
        print("\n" + "=" * 70)
        print("TUNING: TabPFN Causal (feature filter + threshold)")
        print("=" * 70)
        self._ensure_tabpfn_token()

        try:
            parents = CausalDiscoverer.load_parents(bundle, self.config_path)
        except FileNotFoundError:
            print("  Causal artifacts missing — running discovery first...")
            CausalDiscoverer(self.config_path).discover(bundle)
            parents = CausalDiscoverer.load_parents(bundle, self.config_path)

        X_train_full = bundle.X_train_base[bundle.adm_cols]
        X_val_full = bundle.X_val_base[bundle.adm_cols]
        X_test_full = bundle.X_test_base[bundle.adm_cols]

        def objective(params: Dict[str, Any]) -> float:
            p = self._merge_params(params, self.TABPFN_CAUSAL_DEFAULTS)
            metric = p["threshold_metric"]
            top_k = int(p["top_k_parents"])
            start, stop, step = p["thresh_start"], p["thresh_stop"], p["thresh_step"]
            thresholds = np.arange(start, stop, step)
            scores = []
            for target in bundle.binary_targets[:3]:
                feats = parents.get(target, [])[:top_k]
                X_tr = bundle.X_train_base[feats] if feats else X_train_full
                X_va = bundle.X_val_base[feats] if feats else X_val_full
                model = TabPFNClassifier()
                model.fit(X_tr, bundle.y_train_base[target].astype(int))
                probs = model.predict_proba(X_va)[:, 1]
                y_true = bundle.y_val_base[target].astype(int).values
                best_s = 0.0
                for t in thresholds:
                    pred = (probs >= t).astype(int)
                    s = fbeta_score(y_true, pred, beta=2, zero_division=0) if metric == "f2" else fbeta_score(y_true, pred, beta=1.5, zero_division=0)
                    if s > best_s:
                        best_s = s
                scores.append(best_s)
            return float(np.mean(scores))

        def suggest(trial: optuna.Trial, name: str) -> Any:
            specs = {
                "top_k_parents": lambda: trial.suggest_int("top_k_parents", 3, 15),
                "threshold_metric": lambda: trial.suggest_categorical("threshold_metric", ["f2", "f1.5"]),
                "thresh_start": lambda: trial.suggest_float("thresh_start", 0.0001, 0.01, log=True),
                "thresh_stop": lambda: trial.suggest_float("thresh_stop", 0.3, 0.5),
                "thresh_step": lambda: trial.suggest_categorical("thresh_step", [0.0001, 0.001, 0.01]),
            }
            return specs[name]()

        best = SequentialOptunaTuner(
            ["top_k_parents", "threshold_metric", "thresh_start", "thresh_stop", "thresh_step"],
            suggest, objective,
            n_trials_per_param=max(5, self.n_trials_per_param // 3),
            seed=self.random_state,
        ).optimize()

        test_f2 = self._tabpfn_test(bundle, best, causal=True, parents=parents)
        best["val_score"] = best.pop("_best_score", 0)
        best["test_macro_f2"] = test_f2
        best.pop("_tuning_history", None)
        return best

    def _tabpfn_test(
        self,
        bundle: DataBundle,
        params: Dict[str, Any],
        causal: bool,
        parents: Optional[Dict[str, List[str]]] = None,
    ) -> float:
        metric = params.get("threshold_metric", "f1")
        start = params.get("thresh_start", 0.001)
        stop = params.get("thresh_stop", 0.95)
        step = params.get("thresh_step", 0.001)
        thresholds = np.arange(start, stop, step)
        top_k = int(params.get("top_k_parents", 10))
        scores: List[float] = []

        for target in bundle.binary_targets:
            if causal and parents:
                feats = parents.get(target, [])[:top_k]
                if feats:
                    X_tr = bundle.X_train_base[feats]
                    X_va = bundle.X_val_base[feats]
                    X_te = bundle.X_test_base[feats]
                else:
                    X_tr = bundle.X_train_base[bundle.adm_cols]
                    X_va = bundle.X_val_base[bundle.adm_cols]
                    X_te = bundle.X_test_base[bundle.adm_cols]
            else:
                X_tr = bundle.X_train_base[bundle.adm_cols]
                X_va = bundle.X_val_base[bundle.adm_cols]
                X_te = bundle.X_test_base[bundle.adm_cols]

            model = TabPFNClassifier()
            model.fit(X_tr, bundle.y_train_base[target].astype(int))
            val_probs = model.predict_proba(X_va)[:, 1]
            test_probs = model.predict_proba(X_te)[:, 1]
            y_val = bundle.y_val_base[target].astype(int).values
            y_test = bundle.y_test_base[target].astype(int).values

            best_t = 0.5
            best_s = 0.0
            for t in thresholds:
                pred = (val_probs >= t).astype(int)
                if metric == "f1":
                    s = f1_score(y_val, pred, zero_division=0)
                elif metric == "f1.5":
                    s = fbeta_score(y_val, pred, beta=1.5, zero_division=0)
                else:
                    s = fbeta_score(y_val, pred, beta=2, zero_division=0)
                if s > best_s:
                    best_s, best_t = s, t
            scores.append(fbeta_score(y_test, (test_probs >= best_t).astype(int), beta=2, zero_division=0))
        return float(np.mean(scores))

    def _save_test_results(self, results: Dict[str, Any]) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"Tuning results saved: {self.results_path}")

    def _print_summary(self, results: Dict[str, Any]) -> None:
        print("\n" + "=" * 70)
        print("HYPERPARAMETER TUNING SUMMARY (test set macro-F2 where applicable)")
        print("=" * 70)
        for model, params in results.items():
            test_key = next((k for k in params if k.startswith("test_")), None)
            test_val = params.get(test_key, "N/A") if test_key else "N/A"
            print(f"  {model:20s} -> test score: {test_val}")
