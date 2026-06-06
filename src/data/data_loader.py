"""Centralized UCI data loading, temporal masking, and train/val/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


@dataclass(frozen=True)
class DataBundle:
    """Immutable container for all dataset splits and column metadata."""

    X_full: pd.DataFrame
    y_full: pd.DataFrame
    X_train_base: pd.DataFrame
    X_test_base: pd.DataFrame
    y_train_base: pd.DataFrame
    y_test_base: pd.DataFrame
    X_tune_train: pd.DataFrame
    X_tune_val: pd.DataFrame
    y_tune_train: pd.DataFrame
    y_tune_val: pd.DataFrame
    categorical_cols: List[str]
    numeric_cols: List[str]
    target_names: List[str]
    binary_targets: List[str]
    multiclass_target: str
    day_1_cols: List[str]
    day_2_cols: List[str]
    day_3_cols: List[str]
    future_cols: List[str]
    adm_cols: List[str]

    @property
    def X_val_base(self) -> pd.DataFrame:
        """Alias used by TabPFN runners (same as tune validation split)."""
        return self.X_tune_val

    @property
    def y_val_base(self) -> pd.DataFrame:
        return self.y_tune_val

    def get_admission_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return admission-only features (no future day columns)."""
        drop_cols = self.day_1_cols + self.day_2_cols + self.day_3_cols
        return X.drop(columns=[c for c in drop_cols if c in X.columns]).copy()


class DataLoader:
    """Fetch, preprocess, and split the UCI Myocardial Infarction dataset."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        with open(config_path, encoding="utf-8") as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh)

        self.uci_id: int = self.config["data"]["uci_id"]
        self.test_size: float = self.config["data"]["test_size"]
        self.val_size: float = self.config["data"]["val_size"]
        self.random_state: int = self.config["data"]["random_state"]
        self.cache_dir = Path(self.config["data"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        temporal = self.config["temporal"]
        self.day_1_cols: List[str] = temporal["day_1_cols"]
        self.day_2_cols: List[str] = temporal["day_2_cols"]
        self.day_3_cols: List[str] = temporal["day_3_cols"]

    def load(self) -> DataBundle:
        """Fetch data from UCI, impute targets, and create all splits once."""
        print("Fetching dataset from UCI (id=579)...")
        mi_data = fetch_ucirepo(id=self.uci_id)

        X_full = mi_data.data.features.copy()
        y_full = mi_data.data.targets.copy()
        variables_info = mi_data.variables

        cat_cols_info = variables_info[
            (variables_info["role"] == "Feature") & (variables_info["type"] == "Categorical")
        ]["name"].tolist()
        categorical_cols = [c for c in cat_cols_info if c in X_full.columns]
        numeric_cols = [c for c in X_full.columns if c not in categorical_cols]

        y_full = y_full.fillna(y_full.mode().iloc[0])
        X_full.columns = X_full.columns.astype(str)

        target_names = y_full.columns.tolist()
        binary_targets = target_names[:-1]
        multiclass_target = target_names[-1]

        future_cols = self.day_1_cols + self.day_2_cols + self.day_3_cols + target_names
        adm_cols = [c for c in X_full.columns if c not in future_cols]

        X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
            X_full,
            y_full,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
            X_train_base,
            y_train_base,
            test_size=self.val_size,
            random_state=self.random_state,
        )

        print(
            f"Loaded: {X_full.shape[0]} patients, {X_full.shape[1]} features, "
            f"{y_full.shape[1]} targets."
        )
        print(
            f"Splits -> train: {len(X_train_base)} | val: {len(X_tune_val)} | "
            f"test: {len(X_test_base)}"
        )

        return DataBundle(
            X_full=X_full,
            y_full=y_full,
            X_train_base=X_train_base,
            X_test_base=X_test_base,
            y_train_base=y_train_base,
            y_test_base=y_test_base,
            X_tune_train=X_tune_train,
            X_tune_val=X_tune_val,
            y_tune_train=y_tune_train,
            y_tune_val=y_tune_val,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            target_names=target_names,
            binary_targets=binary_targets,
            multiclass_target=multiclass_target,
            day_1_cols=self.day_1_cols,
            day_2_cols=self.day_2_cols,
            day_3_cols=self.day_3_cols,
            future_cols=future_cols,
            adm_cols=adm_cols,
        )

    @staticmethod
    def generate_temporal_datasets(X_data: pd.DataFrame, bundle: DataBundle) -> Dict[str, pd.DataFrame]:
        """Generate four timeline datasets with future columns masked."""
        day_1 = bundle.day_1_cols
        day_2 = bundle.day_2_cols
        day_3 = bundle.day_3_cols

        datasets: Dict[str, pd.DataFrame] = {}
        drop_admission = day_1 + day_2 + day_3
        datasets["admission"] = X_data.drop(columns=[c for c in drop_admission if c in X_data.columns])

        drop_day_1 = day_2 + day_3
        datasets["day_1"] = X_data.drop(columns=[c for c in drop_day_1 if c in X_data.columns])

        drop_day_2 = day_3
        datasets["day_2"] = X_data.drop(columns=[c for c in drop_day_2 if c in X_data.columns])

        datasets["day_3"] = X_data.copy()
        return datasets

    @staticmethod
    def create_augmented_dataset(
        X_base: pd.DataFrame,
        y_base: pd.DataFrame,
        bundle: DataBundle,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Concatenate four timeline stages into a single augmented dataset."""
        temporal_dicts = DataLoader.generate_temporal_datasets(X_base, bundle)
        X_list: List[pd.DataFrame] = []
        y_list: List[pd.DataFrame] = []
        stages = {"admission": 0, "day_1": 1, "day_2": 2, "day_3": 3}

        for stage_name, df_stage in temporal_dicts.items():
            df_copy = df_stage.copy()
            df_copy["TIMELINE_STAGE"] = stages[stage_name]
            X_list.append(df_copy)
            y_list.append(y_base.copy())

        X_aug = pd.concat(X_list, ignore_index=True)
        y_aug = pd.concat(y_list, ignore_index=True)
        return X_aug, y_aug

    @staticmethod
    def fix_categorical_types(df: pd.DataFrame, cat_columns: List[str]) -> pd.DataFrame:
        """Ensure categorical columns are properly typed for tree/NN models."""
        df = df.copy()
        for col in cat_columns:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .replace({"nan": "Unknown", "NaN": "Unknown"})
                    .astype("category")
                )
        if "TIMELINE_STAGE" in df.columns:
            df["TIMELINE_STAGE"] = df["TIMELINE_STAGE"].astype("category")
        return df

    def cache_raw(self, bundle: DataBundle) -> None:
        """Optionally persist raw splits to parquet for reproducibility."""
        bundle.X_full.to_parquet(self.cache_dir / "X_full.parquet")
        bundle.y_full.to_parquet(self.cache_dir / "y_full.parquet")
