"""
PredicateBagger: bootstrap/subsample predicates across multiple bags for
robust metric estimation.
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


class PredicateBagger:
    """Perform predicate bagging with granular control over sampling strategy.

    Bagging creates repeated random subsets of samples and/or predicates,
    evaluating each predicate on the subset.  This yields a distribution of
    predicate coverage that is used downstream to compute robust association
    metrics (see :mod:`smx.predicates.metrics`).

    Parameters
    ----------
    n_bags : int, default 50
        Number of bags (iterations) to create.
    n_predicates_per_bag : int, default 20
        Number of predicates to draw per bag (ignored when
        ``predicate_bagging=False``).
    n_samples_per_bag : int, default 80
        Number of samples to draw per bag (ignored when
        ``sample_bagging=False``).
    min_samples_per_predicate : int, default 5
        Minimum samples satisfying a predicate for it to be included in a bag.
        Applied only when ``sample_bagging=True``.
    replace : bool, default True
        Whether to sample with replacement (bootstrap).  Ignored when
        ``sample_bagging=False``.
    random_seed : int, default 42
        Base random seed for reproducibility.
    sample_bagging : bool, default True
        If ``False``, all samples are used in every bag.
    predicate_bagging : bool, default True
        If ``False``, all predicates are used in every bag.
    """

    def __init__(
        self,
        n_bags: int = 50,
        n_predicates_per_bag: int = 20,
        n_samples_per_bag: int = 80,
        min_samples_per_predicate: int = 5,
        replace: bool = True,
        random_seed: int = 42,
        sample_bagging: bool = True,
        predicate_bagging: bool = True,
    ) -> None:
        self.n_bags = n_bags
        self.n_predicates_per_bag = n_predicates_per_bag
        self.n_samples_per_bag = n_samples_per_bag
        self.min_samples_per_predicate = min_samples_per_predicate
        self.replace = replace
        self.random_seed = random_seed
        self.sample_bagging = sample_bagging
        self.predicate_bagging = predicate_bagging

    def run(
        self,
        zone_scores_df: pd.DataFrame,
        y_predicted_numeric: Union[pd.Series, np.ndarray],
        predicates_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Create bags by sampling samples and/or predicates.

        Parameters
        ----------
        zone_scores_df : pd.DataFrame
            Aggregated zone scores (samples × zones).
        y_predicted_numeric : pd.Series or np.ndarray
            Continuous model predictions aligned with *zone_scores_df*.
        predicates_df : pd.DataFrame
            Predicate catalogue with columns ``rule``, ``zone``,
            ``thresholds``, ``operator``.

        Returns
        -------
        dict
            ``{'Bag_1': {rule: DataFrame(['Zone_Sum', 'Predicted_Y',
            'Sample_Index']), ...}, 'Bag_2': ...}``
        """
        if isinstance(y_predicted_numeric, np.ndarray):
            y_predicted_numeric = pd.Series(y_predicted_numeric)

        np.random.seed(self.random_seed)

        n_total = len(zone_scores_df)
        all_rules = predicates_df["rule"].tolist()
        bags: Dict[str, Dict[str, pd.DataFrame]] = {}

        for bag_num in range(1, self.n_bags + 1):
            # ── Sample selection ──────────────────────────────────────────
            if self.sample_bagging:
                bag_indices = np.random.choice(
                    range(n_total),
                    size=self.n_samples_per_bag,
                    replace=self.replace,
                )
            else:
                bag_indices = np.arange(n_total)

            # ── Predicate selection ───────────────────────────────────────
            if self.predicate_bagging:
                selected_rules = np.random.choice(
                    all_rules,
                    size=min(self.n_predicates_per_bag, len(all_rules)),
                    replace=False,
                )
            else:
                selected_rules = all_rules

            # ── Build bag ────────────────────────────────────────────────
            bag_predicates: Dict[str, pd.DataFrame] = {}
            n_discarded = 0

            for rule in selected_rules:
                rows = predicates_df[predicates_df["rule"] == rule]
                if rows.empty:
                    continue
                pred_row = rows.iloc[0]
                zone = pred_row["zone"]
                threshold = float(pred_row["thresholds"])
                operator = pred_row["operator"]

                zone_vals = zone_scores_df.loc[bag_indices, zone].values
                if operator == "<=":
                    mask = zone_vals <= threshold
                elif operator == ">":
                    mask = zone_vals > threshold
                else:
                    continue

                satisfied = bag_indices[mask]

                if self.sample_bagging and len(satisfied) < self.min_samples_per_predicate:
                    n_discarded += 1
                    continue
                if len(satisfied) == 0:
                    n_discarded += 1
                    continue

                bag_predicates[rule] = pd.DataFrame({
                    "Zone_Sum": zone_scores_df.loc[satisfied, zone].values,
                    "Predicted_Y": y_predicted_numeric.iloc[satisfied].values,
                    "Sample_Index": satisfied,
                })

            if bag_predicates:
                bags[f"Bag_{bag_num}"] = bag_predicates
                samp_str = "yes" if self.sample_bagging else "no"
                pred_str = (
                    f"yes ({self.n_predicates_per_bag})"
                    if self.predicate_bagging
                    else "no (all)"
                )
                print(
                    f"Bag_{bag_num} | samples: {samp_str} | predicates: {pred_str} | "
                    f"valid: {len(bag_predicates)} | discarded: {n_discarded}"
                )
            else:
                print(f"Bag_{bag_num}: EMPTY (all predicates discarded)")

        return bags
