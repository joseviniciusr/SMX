"""
PredicateGenerator: generate binary predicates from quantile thresholds on
zone aggregation scores.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


class PredicateGenerator:
    """Generate binary predicates from quantile thresholds on zone scores.

    For each spectral zone and each quantile value, two predicates are created:

    * ``zone <= q_value`` (samples below the quantile threshold)
    * ``zone > q_value``  (samples above the quantile threshold)

    Duplicate predicates (same rule) arising from identical quantile values
    are automatically removed.

    Parameters
    ----------
    quantiles : list of float
        Quantile fractions in [0, 1] to use as thresholds.
        Example: ``[0.25, 0.5, 0.75]`` creates six predicates per zone.

    Attributes (set after :meth:`fit`)
    ------------------------------------
    predicates_df_ : pd.DataFrame
        One row per predicate. Columns: ``predicate``, ``rule``, ``zone``,
        ``thresholds``, ``operator``.
    indicator_df_ : pd.DataFrame
        Binary indicator matrix (samples × predicates).  Columns are predicate
        rule strings; values are 1/0.
    co_occurrence_matrix_ : pd.DataFrame
        Pairwise co-occurrence matrix of predicates.
    """

    def __init__(self, quantiles: List[float]) -> None:
        if not quantiles:
            raise ValueError("quantiles must be a non-empty list.")
        self.quantiles = quantiles
        self.predicates_df_: Optional[pd.DataFrame] = None
        self.indicator_df_: Optional[pd.DataFrame] = None
        self.co_occurrence_matrix_: Optional[pd.DataFrame] = None
        self._zone_quantile_values_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, zone_scores_df: pd.DataFrame) -> "PredicateGenerator":
        """Learn predicates from *zone_scores_df*.

        Parameters
        ----------
        zone_scores_df : pd.DataFrame
            Zone aggregation scores (samples × zones) as returned by
            :class:`smx.zones.aggregation.ZoneAggregator`.

        Returns
        -------
        self
        """
        self._zone_quantile_values_ = zone_scores_df.quantile(self.quantiles)

        predicate_rows = []
        predicate_num = 1
        for zone in zone_scores_df.columns:
            for q in self.quantiles:
                q_value = self._zone_quantile_values_.loc[q, zone]
                predicate_rows.append({
                    "predicate": f"P{predicate_num}",
                    "rule": f"{zone} <= {q_value:.2f}",
                    "zone": zone,
                    "thresholds": f"{q_value:.2f}",
                    "operator": "<=",
                })
                predicate_num += 1
                predicate_rows.append({
                    "predicate": f"P{predicate_num}",
                    "rule": f"{zone} > {q_value:.2f}",
                    "zone": zone,
                    "thresholds": f"{q_value:.2f}",
                    "operator": ">",
                })
                predicate_num += 1

        predicates_df = pd.DataFrame(predicate_rows)

        # Remove duplicate rules (can appear when multiple quantiles collapse to
        # the same value, e.g. highly discrete data).
        n_before = len(predicates_df)
        predicates_df = (
            predicates_df.drop_duplicates(subset=["rule"], keep="first")
            .reset_index(drop=True)
        )
        n_after = len(predicates_df)
        if n_before != n_after:
            print(
                f"Removed {n_before - n_after} duplicate predicates. "
                f"Remaining: {n_after}"
            )

        # Re-number predicates after deduplication.
        predicates_df["predicate"] = [f"P{i + 1}" for i in range(len(predicates_df))]

        self.predicates_df_ = predicates_df
        self.indicator_df_, self.co_occurrence_matrix_ = self._build_indicator(
            zone_scores_df, predicates_df
        )
        return self

    def transform(self, zone_scores_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted predicates to new zone scores data.

        This re-uses the thresholds learned during :meth:`fit` and is
        useful for applying predicates to a prediction (validation) set.

        Parameters
        ----------
        zone_scores_df : pd.DataFrame
            Zone scores for the samples to evaluate.

        Returns
        -------
        pd.DataFrame
            Binary indicator matrix (samples × predicates).
        """
        if self.predicates_df_ is None:
            raise RuntimeError("Call fit() before transform().")
        indicator_df, _ = self._build_indicator(zone_scores_df, self.predicates_df_)
        return indicator_df

    def fit_transform(self, zone_scores_df: pd.DataFrame) -> pd.DataFrame:
        """Fit and return the indicator matrix in one step."""
        self.fit(zone_scores_df)
        return self.indicator_df_

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_indicator(
        zone_scores_df: pd.DataFrame,
        predicates_df: pd.DataFrame,
    ):
        """Compute the binary indicator matrix and co-occurrence matrix."""

        def _eval(value: float, threshold: str, operator: str) -> int:
            t = float(threshold)
            if operator == "<=":
                return int(value <= t)
            elif operator == ">":
                return int(value > t)
            return 0

        columns_dict = {}
        for _, row in predicates_df.iterrows():
            zone = row["zone"]
            if zone not in zone_scores_df.columns:
                continue
            columns_dict[row["rule"]] = zone_scores_df[zone].apply(
                lambda v, t=row["thresholds"], op=row["operator"]: _eval(v, t, op)
            )

        indicator_df = pd.DataFrame(columns_dict, index=zone_scores_df.index)

        co_mat = np.dot(indicator_df.T, indicator_df)
        co_occurrence_matrix = pd.DataFrame(
            co_mat, index=indicator_df.columns, columns=indicator_df.columns
        )
        return indicator_df, co_occurrence_matrix
