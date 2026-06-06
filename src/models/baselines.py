"""Baseline models: XGBoost, PyTorch multi-task NN, and AutoGluon classifier chain."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import yaml
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, fbeta_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

from src.data.data_loader import DataBundle, DataLoader
from src.evaluation.metrics import optimize_threshold, print_section_header, print_target_results
from src.tuning.hparams_store import load_best_hyperparams
from src.utils.device import device_summary, get_device, get_xgboost_device, setup_cuda


from src.models.pytorch_net import MultiTaskMI


class BaselineRunner:
    """Orchestrate XGBoost, PyTorch NN, and AutoGluon baseline pipelines."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        with open(config_path, encoding="utf-8") as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh)

        self.random_state = self.config["data"]["random_state"]
        thresh = self.config["thresholds"]
        self.min_recall = thresh["min_recall_acceptable"]
        xgb_thresh = thresh["xgboost"]
        self.xgb_thresholds = np.arange(xgb_thresh["start"], xgb_thresh["stop"], xgb_thresh["step"])

        self.xgb_models: Dict[str, Tuple[xgb.XGBClassifier, float]] = {}
        self.pytorch_model: Optional[MultiTaskMI] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.autogluon_predictors: Dict[str, TabularPredictor] = {}
        self.best_hp = load_best_hyperparams()
        setup_cuda()
        print(device_summary())

    def run_all(self, bundle: DataBundle) -> None:
        self.run_xgboost(bundle)
        self.run_pytorch(bundle)
        self.run_autogluon(bundle)

    def run_xgboost(self, bundle: DataBundle) -> Dict[str, Tuple[xgb.XGBClassifier, float]]:
        """XGBoost with grid search, early stopping, and recall-constrained thresholds."""
        print_section_header("XGBOOST BASELINE — Admission Features")

        X_adm_train = bundle.get_admission_features(bundle.X_train_base).apply(pd.to_numeric, errors="coerce")
        X_adm_test = bundle.get_admission_features(bundle.X_test_base).apply(pd.to_numeric, errors="coerce")
        X_tune_train = bundle.get_admission_features(bundle.X_tune_train).apply(pd.to_numeric, errors="coerce")
        X_tune_val = bundle.get_admission_features(bundle.X_tune_val).apply(pd.to_numeric, errors="coerce")

        xgb_cfg = self.config["xgboost"]
        hp = self.best_hp.get("xgboost", {})
        tree_method, device = get_xgboost_device()

        if hp:
            best_params = {k: hp[k] for k in [
                "max_depth", "learning_rate", "subsample", "colsample_bytree",
                "min_child_weight", "reg_alpha", "reg_lambda",
            ] if k in hp}
            print(f"Using tuned XGBoost params: {best_params} | device={device}")
        else:
            best_params = None
            depths = xgb_cfg["depths"]
            learning_rates = xgb_cfg["learning_rates"]

        for target in bundle.target_names:
            print(f"\n--- Processing target: {target} ---")

            y_train = bundle.y_train_base[target].astype(int)
            y_test = bundle.y_test_base[target].astype(int)
            y_tune_train = bundle.y_tune_train[target].astype(int)
            y_tune_val = bundle.y_tune_val[target].astype(int)

            weights_tune = compute_sample_weight("balanced", y_tune_train)
            weights_base = compute_sample_weight("balanced", y_train)

            if best_params is None:
                best_val_score = -1.0
                best_params_target = {"max_depth": 5, "learning_rate": 0.05}

                for depth in depths:
                    for lr in learning_rates:
                        search_params = {
                            "n_estimators": xgb_cfg["n_estimators_search"],
                            "max_depth": depth,
                            "learning_rate": lr,
                            "subsample": xgb_cfg["subsample"],
                            "colsample_bytree": xgb_cfg["colsample_bytree"],
                            "tree_method": tree_method,
                            "device": device,
                            "random_state": self.random_state,
                            "n_jobs": -1,
                        }
                        model_search = xgb.XGBClassifier(**search_params)
                        model_search.fit(X_tune_train, y_tune_train, sample_weight=weights_tune)
                        preds = model_search.predict(X_tune_val)

                        if target in bundle.binary_targets:
                            score = fbeta_score(y_tune_val, preds, beta=2, zero_division=0)
                        else:
                            score = accuracy_score(y_tune_val, preds)

                        if score > best_val_score:
                            best_val_score = score
                            best_params_target = {"max_depth": depth, "learning_rate": lr}
                target_params = best_params_target
            else:
                target_params = best_params

            print(f"Params: depth={target_params['max_depth']}, lr={target_params['learning_rate']}, device={device}")

            tune_params = {
                "n_estimators": xgb_cfg["n_estimators_tune"],
                "max_depth": int(target_params["max_depth"]),
                "learning_rate": target_params["learning_rate"],
                "subsample": target_params.get("subsample", xgb_cfg["subsample"]),
                "colsample_bytree": target_params.get("colsample_bytree", xgb_cfg["colsample_bytree"]),
                "min_child_weight": int(target_params.get("min_child_weight", 1)),
                "reg_alpha": target_params.get("reg_alpha", 0.0),
                "reg_lambda": target_params.get("reg_lambda", 1.0),
                "tree_method": tree_method,
                "device": device,
                "random_state": self.random_state,
                "n_jobs": -1,
                "early_stopping_rounds": xgb_cfg["early_stopping_rounds"],
                "eval_metric": "aucpr" if target in bundle.binary_targets else "mlogloss",
            }

            model_tune = xgb.XGBClassifier(**tune_params)
            model_tune.fit(
                X_tune_train,
                y_tune_train,
                sample_weight=weights_tune,
                eval_set=[(X_tune_val, y_tune_val)],
                verbose=False,
            )

            optimal_trees = model_tune.best_iteration
            print(f"Optimal trees: {optimal_trees}")

            final_threshold = 0.50
            if target in bundle.binary_targets:
                val_probs = model_tune.predict_proba(X_tune_val)[:, 1]
                final_threshold, _, _ = optimize_threshold(
                    y_tune_val.values,
                    val_probs,
                    self.xgb_thresholds,
                    metric="recall_constrained",
                    min_recall=self.min_recall,
                    beta=2.0,
                )
                print(f"Scientific threshold: {final_threshold:.3f}")

            refit_params = tune_params.copy()
            refit_params["n_estimators"] = optimal_trees
            refit_params.pop("early_stopping_rounds", None)
            refit_params.pop("eval_metric", None)

            model_final = xgb.XGBClassifier(**refit_params)
            model_final.fit(X_adm_train, y_train, sample_weight=weights_base)
            self.xgb_models[target] = (model_final, final_threshold)

            if target in bundle.binary_targets:
                test_probs = model_final.predict_proba(X_adm_test)[:, 1]
                y_pred = (test_probs >= final_threshold).astype(int)
                print_target_results(target, y_test.values, y_pred, threshold=final_threshold)
            else:
                y_pred = model_final.predict(X_adm_test)
                print_target_results(target, y_test.values, y_pred, is_binary=False)

        return self.xgb_models

    def run_pytorch(self, bundle: DataBundle) -> MultiTaskMI:
        """Multi-task PyTorch NN with temporal augmentation and early stopping."""
        print_section_header("PYTORCH MULTI-TASK NN — Temporal Augmentation")

        pt_cfg = self.config["pytorch"]
        hp = self.best_hp.get("pytorch", {})
        device = get_device()
        loader = DataLoader()

        X_train_aug, y_train_aug = loader.create_augmented_dataset(
            bundle.X_train_base, bundle.y_train_base, bundle
        )

        cat_cols_nn = bundle.categorical_cols + ["TIMELINE_STAGE"]
        num_cols_nn = bundle.numeric_cols

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self.preprocessor = ColumnTransformer([
            ("num", numeric_transformer, num_cols_nn),
            ("cat", categorical_transformer, cat_cols_nn),
        ])

        X_processed = self.preprocessor.fit_transform(X_train_aug)
        y_bin = y_train_aug[bundle.binary_targets].values.astype(np.float32)
        y_multi = y_train_aug[bundle.multiclass_target].values.astype(np.int64)

        X_t, X_v, y_b_t, y_b_v, y_m_t, y_m_v = train_test_split(
            X_processed, y_bin, y_multi, test_size=0.2, random_state=self.random_state
        )

        X_t_tensor = torch.tensor(X_t, dtype=torch.float32, device=device)
        y_b_t_tensor = torch.tensor(y_b_t, dtype=torch.float32, device=device)
        y_m_t_tensor = torch.tensor(y_m_t, dtype=torch.long, device=device)
        X_v_tensor = torch.tensor(X_v, dtype=torch.float32, device=device)
        y_b_v_tensor = torch.tensor(y_b_v, dtype=torch.float32, device=device)
        y_m_v_tensor = torch.tensor(y_m_v, dtype=torch.long, device=device)

        batch_size = int(hp.get("batch_size", pt_cfg["batch_size"]))
        train_loader = DataLoader(
            TensorDataset(X_t_tensor, y_b_t_tensor, y_m_t_tensor),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_v_tensor, y_b_v_tensor, y_m_v_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

        num_positives = y_b_t_tensor.sum(dim=0)
        num_negatives = y_b_t_tensor.shape[0] - num_positives
        pos_weight = (num_negatives / (num_positives + 1e-5)) * pt_cfg["pos_weight_multiplier"]
        pos_weight = pos_weight.to(device)

        criterion_binary = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion_multi = nn.CrossEntropyLoss()
        multi_weight = pt_cfg["multiclass_loss_weight"]

        if hp:
            best_params = {
                "lr": hp["learning_rate"],
                "dropout": hp["dropout"],
                "weight_decay": hp.get("weight_decay", 1e-4),
            }
            print(f"Using tuned PyTorch params: {best_params} | device={device}")
        else:
            best_val_loss = float("inf")
            best_params = {"lr": 0.001, "dropout": 0.3, "weight_decay": 1e-4}

            print("Hyperparameter tuning (grid fallback)...")
            for lr in pt_cfg["learning_rates"]:
                for drop in pt_cfg["dropout_rates"]:
                    tune_model = MultiTaskMI(
                        input_size=X_t_tensor.shape[1],
                        num_binary_targets=len(bundle.binary_targets),
                        num_multiclass_classes=pt_cfg["num_multiclass_classes"],
                        dropout_rate=drop,
                    ).to(device)
                    tune_opt = optim.Adam(tune_model.parameters(), lr=lr, weight_decay=1e-4)

                    for _ in range(pt_cfg["tune_epochs"]):
                        tune_model.train()
                        for batch_X, batch_y_bin, batch_y_multi in train_loader:
                            tune_opt.zero_grad()
                            out_bin, out_multi = tune_model(batch_X)
                            loss = criterion_binary(out_bin, batch_y_bin) + criterion_multi(out_multi, batch_y_multi) * multi_weight
                            loss.backward()
                            tune_opt.step()

                    tune_model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_X, batch_y_bin, batch_y_multi in val_loader:
                            out_bin, out_multi = tune_model(batch_X)
                            loss = criterion_binary(out_bin, batch_y_bin) + criterion_multi(out_multi, batch_y_multi) * multi_weight
                            val_loss += loss.item()

                    avg_val_loss = val_loss / len(val_loader)
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_params = {"lr": lr, "dropout": drop, "weight_decay": 1e-4}

        print(f"Optimal: LR={best_params['lr']}, Dropout={best_params['dropout']}, device={device}")

        self.pytorch_model = MultiTaskMI(
            input_size=X_t_tensor.shape[1],
            num_binary_targets=len(bundle.binary_targets),
            num_multiclass_classes=pt_cfg["num_multiclass_classes"],
            dropout_rate=best_params["dropout"],
        ).to(device)
        optimizer = optim.Adam(
            self.pytorch_model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params.get("weight_decay", 1e-4),
        )

        best_deep_val_loss = float("inf")
        epochs_no_improve = 0
        best_weights = None

        print("Deep training with early stopping...")
        for epoch in range(pt_cfg["max_epochs"]):
            self.pytorch_model.train()
            for batch_X, batch_y_bin, batch_y_multi in train_loader:
                optimizer.zero_grad()
                out_bin, out_multi = self.pytorch_model(batch_X)
                loss = criterion_binary(out_bin, batch_y_bin) + criterion_multi(out_multi, batch_y_multi) * multi_weight
                loss.backward()
                optimizer.step()

            self.pytorch_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y_bin, batch_y_multi in val_loader:
                    out_bin, out_multi = self.pytorch_model(batch_X)
                    loss = criterion_binary(out_bin, batch_y_bin) + criterion_multi(out_multi, batch_y_multi) * multi_weight
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_deep_val_loss:
                best_deep_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_weights = copy.deepcopy(self.pytorch_model.state_dict())
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1:03d}/{pt_cfg['max_epochs']}] | "
                    f"Val Loss: {avg_val_loss:.4f} | Patience: {epochs_no_improve}/{pt_cfg['patience']}"
                )

            if epochs_no_improve >= pt_cfg["patience"]:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

        if best_weights is not None:
            self.pytorch_model.load_state_dict(best_weights)

        self._evaluate_pytorch_admission(bundle)
        return self.pytorch_model

    def _evaluate_pytorch_admission(self, bundle: DataBundle) -> None:
        """Evaluate PyTorch model on admission-only test set."""
        assert self.pytorch_model is not None and self.preprocessor is not None

        loader = DataLoader()
        X_test_adm = loader.generate_temporal_datasets(bundle.X_test_base, bundle)["admission"].copy()
        X_test_adm["TIMELINE_STAGE"] = 0

        expected_cols = self.preprocessor.feature_names_in_.tolist()
        for col in expected_cols:
            if col not in X_test_adm.columns:
                X_test_adm[col] = np.nan
        X_test_adm = X_test_adm[expected_cols]

        X_test_processed = self.preprocessor.transform(X_test_adm)
        device = get_device()
        X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32, device=device)

        self.pytorch_model.eval()
        with torch.no_grad():
            out_bin, _ = self.pytorch_model(X_test_tensor)
            pt_probs = torch.sigmoid(out_bin).cpu().numpy()

        comp_cfg = self.config["thresholds"]["comparison"]
        thresholds = np.arange(comp_cfg["start"], comp_cfg["stop"], comp_cfg["step"])

        for i, target in enumerate(bundle.binary_targets):
            y_true = bundle.y_test_base[target].values.astype(int)
            y_probs = pt_probs[:, i]
            final_thresh, _, _ = optimize_threshold(
                y_true,
                y_probs,
                thresholds,
                metric="recall_constrained",
                min_recall=comp_cfg["min_recall"],
            )
            y_pred = (y_probs >= final_thresh).astype(int)
            print_target_results(target, y_true, y_pred, threshold=final_thresh, extra={"Model": "PyTorch_NN"})

    def run_autogluon(self, bundle: DataBundle) -> Dict[str, TabularPredictor]:
        """AutoGluon classifier chain on admission features."""
        print_section_header("AUTOGLUON CLASSIFIER CHAIN — Admission Features")

        ag_cfg = self.config["autogluon"]
        hp = self.best_hp.get("autogluon", {})
        presets = hp.get("presets", ag_cfg["presets"])
        time_limit = int(hp.get("time_limit", ag_cfg["time_limit"]))
        if hp:
            print(f"Using tuned AutoGluon params: presets={presets}, time_limit={time_limit}")
        base_dir = ag_cfg["models_dir"]

        X_adm = bundle.get_admission_features(bundle.X_full)
        df_full = pd.concat([X_adm, bundle.y_full], axis=1)

        train_data, test_data = train_test_split(
            df_full, test_size=self.config["data"]["test_size"], random_state=self.random_state
        )
        train_data = TabularDataset(train_data)
        test_data = TabularDataset(test_data)

        predicted_train = train_data.copy()
        predicted_test = test_data.copy()

        for i, target in enumerate(bundle.target_names):
            print(f"\n--- Processing target: {target} ({i+1}/{len(bundle.target_names)}) ---")
            model_path = os.path.join(base_dir, target)

            future_targets = bundle.target_names[i + 1:]
            train_input = predicted_train.drop(columns=future_targets).copy()
            test_input = predicted_test.drop(columns=future_targets).copy()

            if os.path.exists(model_path):
                print(f"Loading existing model from {model_path}")
                predictor = TabularPredictor.load(model_path)
            else:
                print(f"Training new model at {model_path}")
                weight_col = None
                prob_type = "multiclass"
                metric = "accuracy"

                if target in bundle.binary_targets:
                    num_pos = (train_input[target] == 1).sum()
                    num_neg = (train_input[target] == 0).sum()
                    weight_ratio = num_neg / (num_pos + 1e-5)
                    train_input["sample_weight"] = np.where(
                        train_input[target] == 1, weight_ratio, 1.0
                    )
                    weight_col = "sample_weight"
                    prob_type = "binary"
                    metric = "roc_auc"

                predictor = TabularPredictor(
                    label=target,
                    eval_metric=metric,
                    problem_type=prob_type,
                    sample_weight=weight_col,
                    path=model_path,
                ).fit(
                    train_input,
                    presets=presets,
                    time_limit=time_limit,
                    excluded_model_types=ag_cfg["excluded_model_types"],
                    verbosity=0,
                )

            self.autogluon_predictors[target] = predictor

            if "sample_weight" in train_input.columns:
                train_input = train_input.drop(columns=["sample_weight"])

            predicted_train[target] = predictor.predict(train_input)
            predicted_test[target] = predictor.predict(test_input)

        self._evaluate_autogluon_chain(bundle)
        print("\nClassifier chain complete.")
        return self.autogluon_predictors

    def _evaluate_autogluon_chain(self, bundle: DataBundle) -> None:
        """Evaluate AutoGluon chain with threshold optimization on test set."""
        loader = DataLoader()
        chain_test = loader.generate_temporal_datasets(bundle.X_test_base, bundle)["admission"].copy()

        comp_cfg = self.config["thresholds"]["comparison"]
        thresholds = np.arange(comp_cfg["start"], comp_cfg["stop"], comp_cfg["step"])

        for i, target in enumerate(bundle.target_names):
            y_true = bundle.y_test_base[target].values.astype(int)

            if target in bundle.binary_targets:
                y_probs = self.autogluon_predictors[target].predict_proba(chain_test).iloc[:, 1].values
                final_thresh, _, _ = optimize_threshold(
                    y_true,
                    y_probs,
                    thresholds,
                    metric="recall_constrained",
                    min_recall=comp_cfg["min_recall"],
                )
                y_pred = (y_probs >= final_thresh).astype(int)
                print_target_results(
                    target, y_true, y_pred, threshold=final_thresh, extra={"Model": "AutoGluon_Chain"}
                )

            chain_test[target] = self.autogluon_predictors[target].predict(chain_test)
