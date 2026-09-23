"""Run legacy faithfulness and NAOPC on an existing SMX dataset/model.

The data loader, preprocessing, split, and model-training functions are
imported from the parent ``SMX_research`` workspace.  The ``smx`` package is
forced to come from this fork, so the comparison uses the existing project
protocol while exercising the new evaluator.

Example::

    PYTHONPATH=/home/sbarbonjr/projects/SMX_research/SMX_naopc:/home/sbarbonjr/projects/SMX_research \
      python experiments/naopc/real_dataset_benchmark.py --dataset bank_notes --model svm
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


FORK_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = Path(
    os.environ.get("SMX_RESEARCH_ROOT", FORK_ROOT.parent)
).resolve()
sys.path.insert(0, str(FORK_ROOT))
sys.path.insert(1, str(RESEARCH_ROOT))

from config import build_effective_config  # noqa: E402
from experiments.run_experiment import (  # noqa: E402
    MODEL_CONFIG,
    extract_y_continuous,
    load_data,
    preprocess,
    train_model,
)
from smx import SMX  # noqa: E402


def run(args: argparse.Namespace) -> dict:
    config = build_effective_config(args.dataset, args.model)
    spectral_cuts = [tuple(cut) for cut in config["spectral_cuts"]]
    seed = int(config.get("seed", 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    X_cal, X_test, y_cal, y_test = load_data(config)
    X_cal_prep, X_test_prep, _ = preprocess(config, X_cal, X_test)
    train_result = train_model(
        args.model, config, X_cal_prep, y_cal, X_test_prep, y_test
    )
    model = MODEL_CONFIG[args.model]["model_extractor"](train_result)
    y_pred_cal = extract_y_continuous(args.model, train_result)

    explainer = SMX(
        spectral_cuts=spectral_cuts,
        quantiles=[0.2, 0.4, 0.6, 0.8],
        n_repetitions=args.n_repetitions,
        n_bags=args.n_bags,
        n_samples_fraction=0.8,
        replace=False,
        metric="perturbation",
        estimator=model,
        perturbation_mode="median",
        perturbation_metric=MODEL_CONFIG[args.model]["perturbation_metric"],
        normalize_by_zone_size=True,
        zone_size_exponent=1.0,
        var_exp=True,
        show_graph_details=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        explainer.fit(X_cal_prep, y_pred_cal, X_cal_natural=X_cal)

    started = time.perf_counter()
    legacy = explainer.evaluate_faithfulness(
        X_test_prep,
        ranking="unique",
        X_reference=X_cal_prep,
        masking_strategy=args.masking_strategy,
        n_random_rankings=args.n_random_rankings,
        random_state=seed,
    )
    legacy_runtime = time.perf_counter() - started

    started = time.perf_counter()
    normalized = explainer.evaluate_normalized_faithfulness(
        X_test_prep,
        ranking="unique",
        X_reference=X_cal_prep,
        masking_strategy=args.masking_strategy,
        normalization=args.normalization,
        beam_size=args.beam_size,
        exact_max_zones=args.exact_max_zones,
    )
    normalized_runtime = time.perf_counter() - started

    label_to_num = {
        label: index for index, label in enumerate(pd.Series(y_cal).unique())
    }
    y_test_numeric = np.asarray([label_to_num[label] for label in y_test])
    y_pred_numeric = np.asarray(model.predict(X_test_prep))
    if args.model == "pls":
        y_pred_numeric = (y_pred_numeric >= 0.5).astype(int)
    accuracy = float(np.mean(y_pred_numeric == y_test_numeric))
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "masking_strategy": args.masking_strategy,
        "seed": seed,
        "n_cal": len(X_cal_prep),
        "n_test": len(X_test_prep),
        "n_features": X_test_prep.shape[1],
        "test_accuracy": accuracy,
        "legacy_auc": float(legacy["auc"]),
        "legacy_auc_normalized": float(legacy["auc_normalized"]),
        "legacy_null_percentile": float(legacy["null_percentile"]),
        "legacy_runtime_seconds": float(legacy_runtime),
        "naopc": float(normalized["naopc"]),
        "aopc": float(normalized["aopc"]),
        "naopc_lower_bound": float(normalized["lower_bound"]),
        "naopc_upper_bound": float(normalized["upper_bound"]),
        "normalization": normalized["normalization"],
        "beam_size": normalized["beam_size"],
        "n_zones": normalized["n_zones"],
        "n_model_queries": normalized["n_model_queries"],
        "naopc_runtime_seconds": float(normalized_runtime),
        "naopc_status": normalized["status"],
    }

    result_dir = FORK_ROOT / "experiments" / "naopc" / "real_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.model}_{args.masking_strategy}"
    (result_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    normalized["per_sample"].to_csv(result_dir / f"{stem}_per_sample.csv", index=False)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", choices=["pls", "mlp", "svm"], default="svm")
    parser.add_argument("--masking-strategy", choices=["zero", "mean", "median"], default="zero")
    parser.add_argument("--normalization", choices=["auto", "exact", "beam"], default="auto")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--exact-max-zones", type=int, default=9)
    parser.add_argument("--n-repetitions", type=int, default=4)
    parser.add_argument("--n-bags", type=int, default=10)
    parser.add_argument("--n-random-rankings", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
