"""
ZoneAggregator: reduce each spectral zone (DataFrame) to a single score per sample.

Supports simple column-wise aggregations (sum, mean, …) and PCA-based
aggregation (PC1 score).  A fit/transform interface ensures that the same
PCA model fitted on calibration data can be applied consistently to
prediction data.
"""

from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


_SIMPLE_AGGREGATORS = {
    "sum": lambda df: df.sum(axis=1),
    "mean": lambda df: df.mean(axis=1),
    "median": lambda df: df.median(axis=1),
    "max": lambda df: df.max(axis=1),
    "min": lambda df: df.min(axis=1),
    "std": lambda df: df.std(axis=1),
    "var": lambda df: df.var(axis=1),
    "extreme": lambda df: df.apply(
        lambda row: row.loc[row.abs().idxmax()] if row.notna().any() else np.nan,
        axis=1,
    ),
}


class ZoneAggregator:
    """Aggregate spectral zones to a single score per sample.

    Parameters
    ----------
    method : str, default ``'pca'``
        Aggregation strategy.

        * ``'pca'``: fit a single-component PCA per zone and use PC1 scores.
          Preserves directional information and maximises explained variance.
        * ``'sum'``, ``'mean'``, ``'median'``, ``'max'``, ``'min'``,
          ``'std'``, ``'var'``, ``'extreme'``: simple column-wise aggregations.

    Attributes (set after :meth:`fit`)
    ------------------------------------
    pca_info_ : dict or None
        ``{zone_name: {'pca_model', 'loadings', 'mean', 'variance_explained', 'columns'}}``
        Only populated when ``method='pca'``.
    is_fitted_ : bool
        ``True`` after :meth:`fit` has been called.
    """

    def __init__(
        self,
        method: str = "pca",
    ) -> None:
        valid = {"pca"} | set(_SIMPLE_AGGREGATORS)
        if method not in valid:
            raise ValueError(
                f"Unknown method '{method}'. Valid options: {sorted(valid)}"
            )
        self.method = method
        self.pca_info_: Optional[Dict] = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, spectral_zones_dict: Dict[str, pd.DataFrame]) -> "ZoneAggregator":
        """Fit the aggregator on calibration zone data.

        For ``method='pca'`` this trains a 1-component PCA per zone and stores
        the models so the same projections can be applied to new data.  For
        simple aggregation methods, fit is a no-op (nothing to learn).

        Parameters
        ----------
        spectral_zones_dict : dict[str, pd.DataFrame]
            Calibration spectral zones as returned by
            :func:`smx.zones.extraction.extract_spectral_zones`.

        Returns
        -------
        self
        """
        if self.method == "pca":
            self.pca_info_ = {}
            for zone_name, zone_df in spectral_zones_dict.items():
                X_zone = zone_df.values.astype(float)
                pca = PCA(n_components=1)
                pca.fit(X_zone)
                self.pca_info_[zone_name] = {
                    "pca_model": pca,
                    "loadings": pca.components_[0],
                    "mean": pca.mean_,
                    "variance_explained": pca.explained_variance_ratio_[0],
                    "columns": zone_df.columns.tolist(),
                }
        self.is_fitted_ = True
        return self

    def transform(self, spectral_zones_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Apply the fitted aggregator to zone data.

        Parameters
        ----------
        spectral_zones_dict : dict[str, pd.DataFrame]
            Spectral zones to transform (same structure as used for fit).

        Returns
        -------
        pd.DataFrame
            Scores DataFrame (samples × zones).  For ``method='pca'`` the
            index is taken from the first zone's DataFrame; for simple methods
            it is the shared index of the input DataFrames.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform().")

        scores: Dict[str, pd.Series] = {}

        if self.method == "pca":
            for zone_name, zone_df in spectral_zones_dict.items():
                if zone_name not in self.pca_info_:
                    raise KeyError(
                        f"Zone '{zone_name}' was not seen during fit. "
                        "Ensure the same zones are used for fit and transform."
                    )
                info = self.pca_info_[zone_name]
                pca: PCA = info["pca_model"]
                X_zone = zone_df.values.astype(float)
                zone_scores = pca.transform(X_zone).flatten()
                scores[zone_name] = pd.Series(zone_scores, index=zone_df.index)
        else:
            agg_fn = _SIMPLE_AGGREGATORS[self.method]
            for zone_name, zone_df in spectral_zones_dict.items():
                scores[zone_name] = agg_fn(zone_df)

        return pd.DataFrame(scores)

    def fit_transform(self, spectral_zones_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Fit and transform in one step (convenience wrapper).

        Parameters
        ----------
        spectral_zones_dict : dict[str, pd.DataFrame]
            Calibration spectral zones.

        Returns
        -------
        pd.DataFrame
            Scores DataFrame (samples × zones).
        """
        return self.fit(spectral_zones_dict).transform(spectral_zones_dict)

    # ------------------------------------------------------------------
    # Informational helpers
    # ------------------------------------------------------------------

    def get_variance_explained(self) -> Optional[Dict[str, float]]:
        """Return per-zone explained variance (PCA method only).

        Returns ``None`` for non-PCA methods.
        """
        if self.method != "pca" or self.pca_info_ is None:
            return None
        return {
            zone: info["variance_explained"]
            for zone, info in self.pca_info_.items()
        }
