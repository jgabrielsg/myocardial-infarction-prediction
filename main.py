#!/usr/bin/env python3
"""CLI orchestrator for myocardial infarction complication prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked as `python main.py`
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Myocardial Infarction Complication Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --run-causal
  python main.py --run-baselines
  python main.py --run-tabpfn free
  python main.py --run-tabpfn causal
  python main.py --run-all
  python main.py --run-tuning
  python main.py --run-tuning --tuning-models xgboost pytorch
        """,
    )
    parser.add_argument(
        "--run-tuning",
        action="store_true",
        help="Run sequential Optuna hyperparameter tuning for all 5 models",
    )
    parser.add_argument(
        "--tuning-models",
        nargs="+",
        choices=["xgboost", "pytorch", "autogluon", "tabpfn_free", "tabpfn_causal", "all"],
        default=["all"],
        help="Subset of models to tune (default: all)",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--run-causal",
        action="store_true",
        help="Discover causal DAG with GOLEM and save adjacency matrix",
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Run XGBoost, PyTorch NN, and AutoGluon classifier chain",
    )
    parser.add_argument(
        "--run-tabpfn",
        choices=["free", "causal"],
        metavar="MODE",
        help="Run TabPFN cloud model: 'free' (all adm features) or 'causal' (DAG-filtered)",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run causal discovery, baselines, and both TabPFN modes sequentially",
    )
    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if not any([args.run_causal, args.run_baselines, args.run_tabpfn, args.run_all, args.run_tuning]):
        parser.print_help()
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    from src.causal.dag_generator import CausalDiscoverer
    from src.data.data_loader import DataLoader
    from src.models.baselines import BaselineRunner
    from src.models.tabpfn_causal import TabPFNCausalRunner
    from src.models.tabpfn_free import TabPFNFreeRunner
    from src.tuning.hparams_store import save_best_hyperparams
    from src.tuning.hyperparameter_tuner import HyperparameterTuner

    loader = DataLoader(config_path)
    bundle = loader.load()

    if args.run_tuning:
        print("\n>>> Running hyperparameter tuning...")
        tuner = HyperparameterTuner(config_path)
        models = args.tuning_models
        if "all" in models:
            tuner.run_all(bundle)
        else:
            results = {}
            if "xgboost" in models:
                results["xgboost"] = tuner.tune_xgboost(bundle)
            if "pytorch" in models:
                results["pytorch"] = tuner.tune_pytorch(bundle)
            if "autogluon" in models:
                results["autogluon"] = tuner.tune_autogluon(bundle)
            if "tabpfn_free" in models:
                results["tabpfn_free"] = tuner.tune_tabpfn_free(bundle)
            if "tabpfn_causal" in models:
                results["tabpfn_causal"] = tuner.tune_tabpfn_causal(bundle)
            save_best_hyperparams(results)
            tuner._print_summary(results)

    if args.run_all or args.run_causal:
        print("\n>>> Running causal discovery...")
        discoverer = CausalDiscoverer(config_path)
        discoverer.discover(bundle)

    if args.run_all or args.run_baselines:
        print("\n>>> Running baselines...")
        baselines = BaselineRunner(config_path)
        baselines.run_all(bundle)

    if args.run_all or args.run_tabpfn == "free":
        print("\n>>> Running TabPFN (free mode)...")
        TabPFNFreeRunner(config_path).run(bundle)

    if args.run_all or args.run_tabpfn == "causal":
        print("\n>>> Running TabPFN (causal mode)...")
        TabPFNCausalRunner(config_path).run(bundle)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
