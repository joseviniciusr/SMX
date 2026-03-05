"""
SHAP value generator for SMX pipeline.

Computes SHAP global importance per energy feature and saves the CSV file
that is consumed by debugging.shap_per_zone().

Usage:
    python experiments/run_shap.py --dataset tomato --model mlp
    python experiments/run_shap.py --dataset all --model all
    python experiments/run_shap.py --dataset all --model all --parallel 3
"""

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
SMX_DIR = WORKSPACE_ROOT / 'smx'
if str(SMX_DIR) not in sys.path:
    sys.path.insert(0, str(SMX_DIR))

from config import load_dataset_config, list_available_datasets
from run_experiment import load_data, preprocess, train_model, MODEL_CONFIG

# ── SHAP predict-function dispatch ───────────────────────────────────────────
# PLS outputs continuous predictions; MLP/SVM output P(positive class).
SHAP_PREDICT_FN = {
    'pls': lambda model: model.predict,
    'mlp': lambda model: lambda x: model.predict_proba(x)[:, 1],
    'svm': lambda model: lambda x: model.predict_proba(x)[:, 1],
}


# ── Core routine ─────────────────────────────────────────────────────────────

def run_shap(dataset, model_name, new_only=False):
    """Compute and save SHAP values for a (dataset, model) combination."""
    print(f"\n{'#'*70}")
    print(f"# SHAP: dataset={dataset}, model={model_name}")
    print(f"{'#'*70}\n")

    # 0. Load config
    config = load_dataset_config(dataset)

    # 0a. Skip if output already exists and --new-only was requested
    if new_only:
        output_path = SCRIPT_DIR / model_name.upper() / dataset / f'shap_{config["name"]}.csv'
        if output_path.exists():
            print(f"Skipping (already exists): {output_path}")
            return

    # 1. Set seed for full reproducibility (same as run_experiment)
    if 'seed' not in config:
        raise ValueError(
            f"Dataset config for '{dataset}' is missing the required 'seed' field."
        )
    seed = config['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 2. Validate model compatibility
    if model_name not in config.get('compatible_models', []):
        print(f"WARNING: model '{model_name}' not compatible with dataset '{dataset}'. Skipping.")
        return

    # 3. Load data
    print("Loading data...")
    Xcalclass, Xpredclass, ycalclass, ypredclass = load_data(config)
    print(f"  Xcal: {Xcalclass.shape}, Xpred: {Xpredclass.shape}")

    # 4. Preprocess
    print("Preprocessing...")
    Xcalclass_prep, Xpredclass_prep, prep_info = preprocess(config, Xcalclass, Xpredclass)

    # 5. Train model (identical to run_experiment to guarantee same weights)
    print(f"Training {model_name}...")
    result = train_model(model_name, config, Xcalclass_prep, ycalclass, Xpredclass_prep, ypredclass)

    # 6. Extract fitted model object
    mc = MODEL_CONFIG[model_name]
    model = mc['model_extractor'](result)

    # 7. Compute SHAP values
    print("Computing SHAP values (KernelExplainer)...")
    predict_fn = SHAP_PREDICT_FN[model_name](model)
    explainer = shap.KernelExplainer(predict_fn, Xcalclass_prep)
    shap_exp = explainer(Xcalclass_prep,n_jobs=20)

    shap_global_importance = pd.DataFrame({
        'energy': Xcalclass_prep.columns,
        'Mean_Abs_SHAP': np.abs(shap_exp.values).mean(axis=0)
    })
    shap_global_importance.sort_values(by='Mean_Abs_SHAP', ascending=False, inplace=True)

    # 8. Save to the same path that run_experiment / debugging.py expects
    output_dir = SCRIPT_DIR / model_name.upper() / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'shap_{config["name"]}.csv'
    shap_global_importance.to_csv(output_path, index=False, sep=';')
    print(f"SHAP values saved to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _run_single_wrapper(args_tuple):
    """Wrapper for subprocess-based parallel execution."""
    dataset, model_name, new_only = args_tuple
    try:
        run_shap(dataset, model_name, new_only=new_only)
        return (dataset, model_name, None)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (dataset, model_name, str(e))


def main():
    parser = argparse.ArgumentParser(description='SMX SHAP Value Generator')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (from JSON config) or "all"')
    parser.add_argument('--model', required=True,
                        choices=['pls', 'mlp', 'svm', 'all'],
                        help='Model type or "all"')
    parser.add_argument('--parallel', type=int, default=1, metavar='N',
                        help='Number of parallel SHAP jobs to run (default: 1, sequential)')
    parser.add_argument('--new-only', action='store_true',
                        help='Skip dataset/model combinations whose shap_*.csv already exists')
    args = parser.parse_args()

    datasets = list_available_datasets() if args.dataset == 'all' else [args.dataset]
    models = ['pls', 'mlp', 'svm'] if args.model == 'all' else [args.model]

    jobs = [(ds, mdl) for ds in datasets for mdl in models]

    if args.parallel <= 1:
        # Sequential execution
        for ds, mdl in jobs:
            try:
                run_shap(ds, mdl, new_only=args.new_only)
            except Exception as e:
                print(f"\nERROR running SHAP for {ds}/{mdl}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Spawn child processes so each job gets its own GIL and memory space
        script = str(Path(__file__).resolve())
        running = {}  # pid -> (process, dataset, model)
        pending = list(jobs)

        def _poll_and_report():
            """Check running processes, report any that have finished."""
            finished = []
            for pid, (proc, ds, mdl) in running.items():
                ret = proc.poll()
                if ret is not None:
                    finished.append(pid)
                    status = 'OK' if ret == 0 else f'FAILED (exit {ret})'
                    print(f"\n[parallel] {ds}/{mdl} finished: {status}")
                    if ret != 0:
                        print(proc.stderr.read())
            for pid in finished:
                running.pop(pid)

        while pending or running:
            # Fill up to --parallel slots
            while pending and len(running) < args.parallel:
                ds, mdl = pending.pop(0)
                print(f"[parallel] Launching {ds}/{mdl}...")
                proc = subprocess.Popen(
                    [sys.executable, script, '--dataset', ds, '--model', mdl]
                    + (['--new-only'] if args.new_only else []),
                    stdout=sys.stdout,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                running[proc.pid] = (proc, ds, mdl)

            # Wait a bit before polling again
            if running:
                import time
                time.sleep(2)
                _poll_and_report()

        print("\n[parallel] All SHAP jobs completed.")


if __name__ == '__main__':
    main()
