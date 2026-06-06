"""Causal DAG discovery with GOLEM/NOTEARS and temporal edge banning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from src.data.data_loader import DataBundle
from src.utils.device import device_summary, setup_cuda


class CausalDiscoverer:
    """Learn a temporal DAG from training data and persist adjacency artifacts."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        with open(config_path, encoding="utf-8") as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh)

        causal_cfg = self.config["causal"]
        self.output_dir = Path(causal_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adj_filename = causal_cfg["adj_matrix_filename"]
        self.parents_filename = causal_cfg["parents_filename"]
        self.algorithm = causal_cfg.get("algorithm", "golem")
        self.golem_cfg = causal_cfg["golem"]
        self.notears_cfg = causal_cfg["notears"]
        self.min_parent_weight = causal_cfg.get("min_parent_weight", 0.01)
        self.top_k_parents = causal_cfg.get("top_k_parents", 10)

        self.adj_matrix: Optional[np.ndarray] = None
        self.weight_matrix: Optional[np.ndarray] = None
        self.col_names: List[str] = []
        self.causal_parents_dict: Dict[str, List[str]] = {}
        self.graph: Optional[nx.DiGraph] = None

    def discover(self, bundle: DataBundle) -> Dict[str, List[str]]:
        """Run causal discovery on training patients and apply temporal mask."""
        setup_cuda()
        print(device_summary())

        targets = bundle.binary_targets + [bundle.multiclass_target]

        X_train_full = bundle.X_full.loc[bundle.X_train_base.index].copy()
        adm_cols = bundle.adm_cols

        tier_0 = adm_cols
        tier_1 = bundle.day_1_cols
        tier_2 = bundle.day_2_cols
        tier_3 = bundle.day_3_cols + targets

        df_causal = pd.concat([X_train_full, bundle.y_train_base], axis=1)
        df_causal = df_causal.apply(pd.to_numeric, errors="coerce").fillna(0)
        self.col_names = df_causal.columns.tolist()

        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_causal),
            columns=self.col_names,
            index=df_causal.index,
        )

        if self.algorithm == "notears":
            self._run_notears(df_scaled)
        else:
            self._run_golem(df_scaled)

        self._apply_temporal_mask(tier_0, tier_1, tier_2, tier_3)
        self._build_graph()
        self.causal_parents_dict = self._extract_causal_parents(targets, adm_cols)

        self.save()
        return self.causal_parents_dict

    def _run_golem(self, df_scaled: pd.DataFrame) -> None:
        from castle.algorithms import GOLEM

        cfg = self.golem_cfg
        print("Learning causal graph with GOLEM...")
        gl = GOLEM(
            lambda_1=cfg["lambda_1"],
            lambda_2=cfg["lambda_2"],
            num_iter=cfg["num_iter"],
            graph_thres=cfg["graph_thres"],
        )
        gl.learn(df_scaled)
        self.adj_matrix = np.asarray(gl.causal_matrix).copy()

        if hasattr(gl, "weight_matrix") and gl.weight_matrix is not None:
            self.weight_matrix = np.asarray(gl.weight_matrix).copy()
        elif hasattr(gl, "B") and gl.B is not None:
            self.weight_matrix = np.asarray(gl.B).copy()
        else:
            self.weight_matrix = self.adj_matrix.astype(float)

    def _run_notears(self, df_scaled: pd.DataFrame) -> None:
        from castle.algorithms import Notears

        cfg = self.notears_cfg
        print("Learning causal graph with NOTEARS...")
        nt = Notears(w_threshold=cfg["w_threshold"], max_iter=cfg["max_iter"])
        nt.learn(df_scaled)
        self.adj_matrix = nt.causal_matrix.copy()
        self.weight_matrix = self.adj_matrix.astype(float)

    def _ban_edge(self, source: str, target: str, matrix: np.ndarray) -> None:
        """Zero out edges where future variables would cause past variables."""
        if source in self.col_names and target in self.col_names:
            i = self.col_names.index(source)
            j = self.col_names.index(target)
            matrix[i, j] = 0

    def _apply_temporal_mask(
        self,
        tier_0: List[str],
        tier_1: List[str],
        tier_2: List[str],
        tier_3: List[str],
    ) -> None:
        """Apply chronological constraints: future cannot cause past."""
        assert self.adj_matrix is not None

        matrices: List[np.ndarray] = [self.adj_matrix]
        if self.weight_matrix is not None and not np.shares_memory(self.adj_matrix, self.weight_matrix):
            matrices.append(self.weight_matrix)

        for matrix in matrices:
            for c1 in tier_1:
                for c0 in tier_0:
                    self._ban_edge(c1, c0, matrix)

            for c2 in tier_2:
                for c0 in tier_0 + tier_1:
                    self._ban_edge(c2, c0, matrix)

            for c3 in tier_3:
                for c0 in tier_0 + tier_1 + tier_2:
                    self._ban_edge(c3, c0, matrix)

        print("Temporal edge mask applied (ban_edge).")

    def _build_graph(self) -> None:
        assert self.adj_matrix is not None
        self.graph = nx.DiGraph()

        for i, col_i in enumerate(self.col_names):
            self.graph.add_node(col_i)
            for j, col_j in enumerate(self.col_names):
                if self.adj_matrix[i, j] == 1:
                    self.graph.add_edge(col_i, col_j)

        print(f"Causal graph built: {len(self.graph.edges())} edges.")

    def _extract_causal_parents(
        self,
        targets: List[str],
        adm_cols: List[str],
    ) -> Dict[str, List[str]]:
        """Extract admission-only causal parents for each target."""
        parents_dict: Dict[str, List[str]] = {}

        if self.weight_matrix is not None:
            for target in targets:
                if target not in self.col_names:
                    parents_dict[target] = []
                    continue

                target_idx = self.col_names.index(target)
                parent_weights: List[tuple[str, float]] = []

                for j, col_j in enumerate(self.col_names):
                    weight = abs(float(self.weight_matrix[j, target_idx]))
                    if weight > self.min_parent_weight and col_j in adm_cols:
                        parent_weights.append((col_j, weight))

                parent_weights.sort(key=lambda x: x[1], reverse=True)
                parents_dict[target] = [pw[0] for pw in parent_weights[: self.top_k_parents]]
        elif self.graph is not None:
            for target in targets:
                parents = [
                    edge[0]
                    for edge in self.graph.edges()
                    if edge[1] == target and edge[0] in adm_cols
                ]
                parents_dict[target] = parents
        else:
            for target in targets:
                parents_dict[target] = []

        return parents_dict

    def save(self, plot: bool = True) -> None:
        """Persist adjacency matrix CSV, parents JSON, and optional DAG plot."""
        assert self.adj_matrix is not None

        adj_path = self.output_dir / self.adj_filename
        df_adj = pd.DataFrame(self.adj_matrix, index=self.col_names, columns=self.col_names)
        df_adj.to_csv(adj_path)
        print(f"Adjacency matrix saved: {adj_path}")

        parents_path = self.output_dir / self.parents_filename
        with open(parents_path, "w", encoding="utf-8") as fh:
            json.dump(self.causal_parents_dict, fh, indent=2)
        print(f"Causal parents saved: {parents_path}")

        if plot and self.graph is not None:
            fig_path = self.output_dir / "causal_dag.png"
            plt.figure(figsize=(20, 15))
            nx.draw_networkx(
                self.graph,
                with_labels=True,
                node_size=400,
                node_color="lightblue",
                font_size=7,
                alpha=0.8,
                arrows=True,
            )
            plt.title("Temporal DAG — GOLEM with ban_edge constraints")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"DAG plot saved: {fig_path}")

    @classmethod
    def load_parents(
        cls,
        bundle: DataBundle,
        config_path: str | Path = "configs/config.yaml",
    ) -> Dict[str, List[str]]:
        """Load causal parents from saved JSON, or rebuild from adjacency CSV."""
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        output_dir = Path(config["causal"]["output_dir"])
        parents_path = output_dir / config["causal"]["parents_filename"]
        adj_path = output_dir / config["causal"]["adj_matrix_filename"]

        if parents_path.exists():
            with open(parents_path, encoding="utf-8") as fh:
                return json.load(fh)

        if not adj_path.exists():
            raise FileNotFoundError(
                f"Causal artifacts not found. Run `python main.py --run-causal` first."
            )

        df_adj = pd.read_csv(adj_path, index_col=0)
        targets = bundle.target_names
        parents_dict: Dict[str, List[str]] = {}

        for target in targets:
            if target in df_adj.columns:
                all_parents = df_adj.index[df_adj[target] == 1].tolist()
                parents_dict[target] = [p for p in all_parents if p in bundle.adm_cols]
            else:
                parents_dict[target] = []

        return parents_dict
