import warnings

import numpy as np
import pandas as pd
import pytest

go = pytest.importorskip("plotly.graph_objects")

from sklearn.linear_model import LogisticRegression

import smx
from smx import (
    SMX,
    SMXTheme,
    building_spectral_zones,
    generate_synthetic_spectral_data,
    plot_all_thresholds_overlay,
    plot_faithfulness_curve,
    plot_lrc_bar,
    plot_predicate_heatmap,
    plot_spectrum_with_zones,
    plot_threshold_spectrum,
    plot_zone_ranking_over_spectrum,
    plot_zone_scores,
)
from smx.plotting._common import normalize_cuts, prepare_zone_ranking_df

CUTS = [("bg 1", 1, 20), ("F1", 20, 45), ("bg 2", 45, 60), ("F2", 60, 85), ("bg 3", 85, 100)]


def _peaks(a1, a2):
    return [
        {"center": 30, "amplitude_mean": a1, "amplitude_std": 0.3, "width_mean": 5, "width_std": 1},
        {"center": 70, "amplitude_mean": a2, "amplitude_std": 0.3, "width_mean": 5, "width_std": 1},
    ]


@pytest.fixture(scope="module")
def fitted():
    config = [
        {"name": "A", "n_samples": 40, "peaks": _peaks(2.0, 1.0), "noise_std": 0.05},
        {"name": "B", "n_samples": 40, "peaks": _peaks(1.0, 1.5), "noise_std": 0.05},
    ]
    df = generate_synthetic_spectral_data(classes_config=config, n_points=100, x_min=1, x_max=100, seed=0)
    X = df.drop(columns=["Class"])
    y = df["Class"].reset_index(drop=True)
    X_prep = X - X.mean()
    model = LogisticRegression(max_iter=500).fit(X_prep, y)
    explainer = SMX(
        spectral_cuts=CUTS,
        quantiles=[0.25, 0.5, 0.75],
        n_repetitions=2,
        n_bags=5,
        metric="perturbation",
        estimator=model,
    )
    explainer.fit(X_prep, pd.Series(model.predict_proba(X_prep)[:, 0]), X_cal_natural=X)
    explainer.evaluate_faithfulness(X_prep, n_random_rankings=10)
    return explainer, X, y


@pytest.fixture(autouse=True)
def no_show(monkeypatch):
    """Plot functions must return figures, never display them."""
    def _show(self, *args, **kwargs):
        raise AssertionError("fig.show() must not be called by smx plotting functions")

    monkeypatch.setattr(go.Figure, "show", _show)


def _all_plots(fitted):
    explainer, X, y = fitted
    return {
        "zone_ranking": (
            lambda **kw: plot_zone_ranking_over_spectrum(
                explainer.lrc_natural_, CUTS, explainer.zones_natural_, **kw
            ),
            pd.DataFrame,
        ),
        "spectrum_with_zones": (lambda **kw: plot_spectrum_with_zones(X, CUTS, **kw), pd.DataFrame),
        "threshold": (
            lambda **kw: plot_threshold_spectrum(
                explainer.lrc_natural_, 0, explainer.zones_natural_, explainer.pca_info_natural_, y, **kw
            ),
            pd.Series,
        ),
        "lrc_bar": (lambda **kw: plot_lrc_bar(explainer.lrc_natural_, **kw), pd.DataFrame),
        "heatmap": (lambda **kw: plot_predicate_heatmap(explainer.lrc_natural_, **kw), pd.DataFrame),
        "zone_scores": (lambda **kw: plot_zone_scores(explainer.zones_natural_, y, **kw), pd.DataFrame),
        "overlay": (
            lambda **kw: plot_all_thresholds_overlay(
                explainer.lrc_natural_, explainer.zones_natural_, explainer.pca_info_natural_, y, CUTS, **kw
            ),
            pd.DataFrame,
        ),
        "faithfulness": (lambda **kw: plot_faithfulness_curve(explainer.faithfulness_, **kw), pd.DataFrame),
    }


PLOTS = ["zone_ranking", "spectrum_with_zones", "threshold", "lrc_bar", "heatmap", "zone_scores", "overlay", "faithfulness"]


@pytest.mark.parametrize("name", PLOTS)
def test_common_contract(fitted, tmp_path, name):
    plot, data_type = _all_plots(fitted)[name]

    fig = plot()
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text

    fig, data = plot(return_df=True)
    assert isinstance(fig, go.Figure)
    assert isinstance(data, data_type) and not data.empty

    assert plot(title="").layout.title.text == ""
    assert plot(title=False).layout.title.text == ""
    assert plot(title="Custom").layout.title.text == "Custom"

    fig = plot(width=640, height=360)
    assert (fig.layout.width, fig.layout.height) == (640, 360)

    out = tmp_path / "nested" / f"{name}.html"
    plot(output_path=out)
    assert out.exists() and out.stat().st_size > 0

    with pytest.raises(ValueError, match="Unsupported output format"):
        plot(output_path=tmp_path / f"{name}.txt")


def test_return_df_pair_renders_both_in_jupyter(fitted, monkeypatch):
    explainer, X, y = fitted
    result = plot_lrc_bar(explainer.lrc_natural_, return_df=True)
    assert isinstance(result, tuple) and len(result) == 2
    fig, data = result

    displayed = []
    import IPython.display

    monkeypatch.setattr(IPython.display, "display", displayed.append)
    result._ipython_display_()
    assert displayed[0] is fig and displayed[1] is data


def test_output_path_is_keyword_only(fitted):
    explainer, X, y = fitted
    with pytest.raises(TypeError):
        plot_lrc_bar(explainer.lrc_natural_, "bar.html")
    with pytest.raises(TypeError, match="keyword-only"):
        plot_zone_scores(explainer.zones_natural_, y, "violin.html")


def test_threshold_deprecated_aliases(fitted):
    explainer, X, y = fitted
    with pytest.warns(FutureWarning, match="spectral_zones_original"):
        old = plot_threshold_spectrum(
            lrc_natural_df=explainer.lrc_natural_,
            row_index=0,
            spectral_zones_original=explainer.zones_natural_,
            pca_info_dict_original=explainer.pca_info_natural_,
            y_labels=y,
        )
    new = plot_threshold_spectrum(
        explainer.lrc_natural_, 0, explainer.zones_natural_, explainer.pca_info_natural_, y
    )
    assert old.to_dict() == new.to_dict()

    with pytest.raises(TypeError, match="both"):
        plot_threshold_spectrum(
            explainer.lrc_natural_, 0, explainer.zones_natural_, explainer.pca_info_natural_, y,
            spectral_zones_original=explainer.zones_natural_,
        )
    with pytest.raises(TypeError, match="unexpected"):
        plot_threshold_spectrum(
            explainer.lrc_natural_, 0, explainer.zones_natural_, explainer.pca_info_natural_, y, colour="red"
        )


def test_threshold_one_trace_per_class_with_positional_labels(fitted):
    explainer, X, y = fitted
    shifted = y.copy()
    shifted.index = shifted.index + 1000  # labels must be aligned by position
    fig, spectrum = plot_threshold_spectrum(
        explainer.lrc_natural_, 0, explainer.zones_natural_, explainer.pca_info_natural_, shifted,
        return_df=True,
    )
    names = [t.name for t in fig.data]
    assert names == ["Class A", "Class B", "Threshold spectrum"]
    zone = explainer.lrc_natural_.iloc[0]["Zone"]
    assert len(spectrum) == explainer.zones_natural_[zone].shape[1]

    no_labels = plot_threshold_spectrum(
        explainer.lrc_natural_, 0, explainer.zones_natural_, explainer.pca_info_natural_
    )
    assert [t.name for t in no_labels.data] == ["Samples", "Threshold spectrum"]

    with pytest.raises(ValueError, match="aligned"):
        plot_threshold_spectrum(
            explainer.lrc_natural_, 0, explainer.zones_natural_, explainer.pca_info_natural_, y[:-1]
        )


def test_explicit_colorscale_overrides_theme(fitted):
    explainer, X, y = fitted
    theme = SMXTheme(colorscale="Blues")
    fig_theme = plot_zone_ranking_over_spectrum(
        explainer.lrc_natural_, CUTS, explainer.zones_natural_, theme=theme
    )
    fig_explicit = plot_zone_ranking_over_spectrum(
        explainer.lrc_natural_, CUTS, explainer.zones_natural_, theme=theme, colorscale="YlOrRd"
    )
    theme_fills = [s.fillcolor for s in fig_theme.layout.shapes if s.type == "rect"]
    explicit_fills = [s.fillcolor for s in fig_explicit.layout.shapes if s.type == "rect"]
    assert theme_fills != explicit_fills


def test_grouped_cuts_are_supported(fitted):
    explainer, X, y = fitted
    grouped_cuts = [
        ("bg 1", 1, 20, "background"),
        ("F1", 20, 45),
        ("bg 2", 45, 60, "background"),
        ("F2", 60, 85),
        ("bg 3", 85, 100, "background"),
    ]
    zones = smx.extract_spectral_zones(X, grouped_cuts)
    ranking = pd.DataFrame({"zone": ["F2", "F1", "background"], "score": [3.0, 2.0, 1.0]})
    fig, ranking_df = plot_zone_ranking_over_spectrum(ranking, grouped_cuts, zones, return_df=True)
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(rects) == len(grouped_cuts)
    background_fills = {rects[i].fillcolor for i in (0, 2, 4)}
    assert len(background_fills) == 1  # all members share the group's score colour

    fig = plot_spectrum_with_zones(X, grouped_cuts)
    assert {t.name for t in fig.data} >= {"Spectral zone", "Background"}


def test_normalize_cuts_formats():
    cuts = normalize_cuts([
        (10, 5),
        ("named", 20, 30),
        ("member", 40, 50, "grp"),
        {"name": "d", "start": 60, "end": 70},
        {"start": 80, "end": 90, "group": "grp"},
    ])
    assert cuts["name"].tolist() == ["10-5", "named", "member", "d", "80-90"]
    assert cuts["zone"].tolist() == ["10-5", "named", "grp", "d", "grp"]
    assert cuts.loc[0, ["start", "end"]].tolist() == [5.0, 10.0]
    with pytest.raises(ValueError):
        normalize_cuts([(1, 2, 3, 4, 5)])


def test_prepare_zone_ranking_keeps_provided_ranks_aligned():
    ranking = pd.DataFrame({"zone": ["a", "b", "c"], "score": [1.0, 3.0, 2.0], "rank": [30, 10, 20]})
    prepared = prepare_zone_ranking_df(ranking)
    assert prepared["zone"].tolist() == ["b", "c", "a"]
    assert prepared["rank"].tolist() == [10, 20, 30]

    lrc = pd.DataFrame({"Zone": ["a", "a", "b", None], "Local_Reaching_Centrality": [1.0, 5.0, 2.0, 9.0]})
    prepared = prepare_zone_ranking_df(lrc)
    assert prepared.to_dict("list") == {"zone": ["a", "b"], "score": [5.0, 2.0], "rank": [1, 2]}


def test_heatmap_orders_thresholds_numerically():
    rows = []
    for i in range(12):
        for op in ("<=", ">"):
            rows.append({"Zone": "Z", "Operator": op, "Threshold_Natural": float(i), "Local_Reaching_Centrality": i + 1.0})
    fig, pivot = plot_predicate_heatmap(pd.DataFrame(rows), return_df=True)
    expected = [f"≤ T{i}" for i in range(1, 13)] + [f"> T{i}" for i in range(1, 13)]
    assert pivot.columns.tolist() == expected
    assert list(fig.data[0].x) == expected


def test_class_color_map():
    theme = SMXTheme()
    colors = theme.class_color_map([0, 1, "A", "zz"], {"0": "#000000"})
    assert colors[0] == "#000000"  # int label matched by its string form
    assert colors["A"] == theme.class_colors["A"]
    assert len(set(colors.values())) == 4  # palette fallbacks avoid colours already used
    assert theme.class_color_map(["x", "y"]) == theme.class_color_map(["x", "y"])


def test_overlay_band_and_mean_share_class_colour(fitted):
    explainer, X, y = fitted
    labels = y.map({"A": "first", "B": "second"})  # not in the theme's class_colors
    fig = plot_all_thresholds_overlay(
        explainer.lrc_natural_, explainer.zones_natural_, explainer.pca_info_natural_, labels, CUTS
    )
    for cls in ("first", "second"):
        band, mean = [t for t in fig.data if t.legendgroup == cls]
        assert band.fillcolor == mean.line.color


def test_zone_scores_accepts_fitted_aggregator(fitted):
    explainer, X, y = fitted
    aggregator = smx.ZoneAggregator(method="mean").fit(explainer.zones_natural_)
    fig, scores = plot_zone_scores(explainer.zones_natural_, y, aggregator=aggregator, return_df=True)
    assert fig.layout.yaxis.title.text == "Zone score (mean)"
    np.testing.assert_allclose(scores["F1"], explainer.zones_natural_["F1"].mean(axis=1))
    with pytest.raises(ValueError, match="fitted"):
        plot_zone_scores(explainer.zones_natural_, y, aggregator=smx.ZoneAggregator())


def test_faithfulness_level_intervals_follow_evaluation_bounds(fitted):
    explainer, X, y = fitted
    text = plot_faithfulness_curve(explainer.faithfulness_).layout.annotations[0].text
    assert "Low: &lt; 60" in text and "Very High: ≥ 95" in text
    bare = plot_faithfulness_curve(
        explainer.faithfulness_, show_faithfulness_level=False, show_summary=False, show_level_intervals=False
    )
    assert not bare.layout.annotations


def test_pipeline_plot_methods_return_figures(fitted, tmp_path):
    explainer, X, y = fitted
    fig = explainer.plot_zone_ranking_over_spectrum(X_natural=X, y_labels=y, title="")
    assert isinstance(fig, go.Figure)
    assert {"Class A", "Class B"} <= {t.name for t in fig.data}
    with pytest.raises(ValueError, match="together"):
        explainer.plot_zone_ranking_over_spectrum(X_natural=X)

    fig, curve = explainer.plot_faithfulness(tmp_path / "f.html", return_df=True)
    assert isinstance(fig, go.Figure) and (tmp_path / "f.html").exists()
    assert curve.equals(explainer.faithfulness_["curve_df"].sort_values("k").reset_index(drop=True))


def test_building_spectral_zones_plotting_flags(fitted, monkeypatch, tmp_path):
    explainer, X, y = fitted
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: shown.append(self))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cuts = building_spectral_zones(X, prominence=0.3)
    assert cuts and not shown

    with pytest.warns(FutureWarning, match="ploting"):
        assert building_spectral_zones(X, prominence=0.3, ploting=True) == cuts
    assert len(shown) == 1

    out = tmp_path / "zones.html"
    building_spectral_zones(X, prominence=0.3, output_path=out)
    assert out.exists() and len(shown) == 1


def test_top_level_exports():
    for name in smx.plotting.__all__:
        assert getattr(smx, name) is getattr(smx.plotting, name)
    assert smx.building_spectral_zones is building_spectral_zones
