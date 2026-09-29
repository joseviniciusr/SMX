"""
ZoneAggregator: reduce each spectral zone (DataFrame) to a single score per sample.

Supports simple column-wise aggregations (sum, mean, …) and PCA-based
aggregation (PC1 score).  A fit/transform interface ensures that the same
PCA model fitted on calibration data can be applied consistently to
prediction data.
"""

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

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

        * ``'pca'``: fit a PCA per zone and use the scores of the selected
          components (PC1 only by default).
          Preserves directional information and maximises explained variance.
        * ``'sum'``, ``'mean'``, ``'median'``, ``'max'``, ``'min'``,
          ``'std'``, ``'var'``, ``'extreme'``: simple column-wise aggregations.
    n_components : int or sequence of int, default 1
        Principal components kept per zone (``method='pca'`` only).  An int
        ``k`` keeps PC1..PCk; a sequence keeps exactly those (1-based) PCs,
        e.g. ``[2]`` for PC2 only.  With the default (PC1 only) score columns
        are named after the zone, as in the original SMX; otherwise each
        column is named ``"<zone> [PC<j>]"``.

    Attributes (set after :meth:`fit`)
    ------------------------------------
    pca_info\_ : dict or None
        ``{score_column: {'pca_model', 'loadings', 'mean', 'variance_explained',
        'columns', 'zone', 'pc'}}`` where ``loadings`` and
        ``variance_explained`` refer to that column's component.
        Only populated when ``method='pca'``.
    score_map\_ : dict or None
        ``{score_column: (zone_name, pc)}``.  ``None`` unless components other
        than PC1 alone were requested.
    is_fitted\_ : bool
        ``True`` after :meth:`fit` has been called.
    """

    def __init__(
        self,
        method: str = "pca",
        n_components: Union[int, Sequence[int]] = 1,
    ) -> None:
        valid = {"pca"} | set(_SIMPLE_AGGREGATORS)
        if method not in valid:
            raise ValueError(
                f"Unknown method '{method}'. Valid options: {sorted(valid)}"
            )
        if isinstance(n_components, (int, np.integer)):
            components = list(range(1, int(n_components) + 1))
        else:
            components = sorted({int(c) for c in n_components})
        if not components or components[0] < 1:
            raise ValueError("n_components must select at least one PC (1-based).")
        self.method = method
        self.n_components = n_components
        self.components_: List[int] = components
        self.pca_info_: Optional[Dict] = None
        self.score_map_: Optional[Dict[str, Tuple[str, int]]] = None
        self.is_fitted_: bool = False

    @property
    def _legacy_naming(self) -> bool:
        return self.components_ == [1]

    def _score_column(self, zone_name: str, pc: int) -> str:
        return zone_name if self._legacy_naming else f"{zone_name} [PC{pc}]"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, spectral_zones_dict: Dict[str, pd.DataFrame]) -> "ZoneAggregator":
        """Fit the aggregator on calibration zone data.

        For ``method='pca'`` this trains a PCA per zone (up to the highest
        requested component) and stores the models so the same projections
        can be applied to new data.  For simple aggregation methods, fit is a
        no-op (nothing to learn).

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
            self.score_map_ = None if self._legacy_naming else {}
            for zone_name, zone_df in spectral_zones_dict.items():
                X_zone = zone_df.values.astype(float)
                n_fit = min(self.components_[-1], *X_zone.shape)
                pca = PCA(n_components=n_fit)
                pca.fit(X_zone)
                for pc in self.components_:
                    if pc > n_fit:
                        continue  # zone too narrow for this component
                    col = self._score_column(zone_name, pc)
                    self.pca_info_[col] = {
                        "pca_model": pca,
                        "loadings": pca.components_[pc - 1],
                        "mean": pca.mean_,
                        "variance_explained": pca.explained_variance_ratio_[pc - 1],
                        "columns": zone_df.columns.tolist(),
                        "zone": zone_name,
                        "pc": pc,
                    }
                    if self.score_map_ is not None:
                        self.score_map_[col] = (zone_name, pc)
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
            fitted_zones = {info["zone"] for info in self.pca_info_.values()}
            for zone_name, zone_df in spectral_zones_dict.items():
                if zone_name not in fitted_zones:
                    raise KeyError(
                        f"Zone '{zone_name}' was not seen during fit. "
                        "Ensure the same zones are used for fit and transform."
                    )
                X_zone = zone_df.values.astype(float)
                projected = None
                for col, info in self.pca_info_.items():
                    if info["zone"] != zone_name:
                        continue
                    if projected is None:
                        projected = info["pca_model"].transform(X_zone)
                    scores[col] = pd.Series(
                        projected[:, info["pc"] - 1], index=zone_df.index
                    )
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
        """Return explained variance per score column (PCA method only).

        Returns ``None`` for non-PCA methods.
        """
        if self.method != "pca" or self.pca_info_ is None:
            return None
        return {
            zone: info["variance_explained"]
            for zone, info in self.pca_info_.items()
        }
