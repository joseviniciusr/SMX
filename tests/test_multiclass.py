"""
Comprehensive test suite for SMX multi-class refactoring.

Tests are organised into five groups mirroring the implementation phases:

1. Phase 1 — CovarianceMetric removal
2. Phase 2 / 2b — y_pred_cal removal and y_class_labels interface
3. Phase 3 — Dynamic terminal nodes
4. Phase 4 — Pipeline integration (end-to-end flows)
5. Phase 5 — Plotting colour utilities

Run with:
    pytest tests/test_multiclass.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_spectral_df(n_samples: int, n_features: int, seed: int = 0) -> pd.DataFrame:
    """Return a DataFrame of synthetic spectral data with numeric column names."""
    rng = np.random.default_rng(seed)
    cols = [str(float(i)) for i in range(n_features)]
    return pd.DataFrame(rng.random((n_samples, n_features)), columns=cols)


def make_cuts(n_features: int, n_zones: int) -> list:
    """Return evenly-spaced spectral cuts covering [0, n_features)."""
    step = n_features / n_zones
    return [
        (f"Zone_{i}", float(i * step), float((i + 1) * step))
        for i in range(n_zones)
    ]


def make_trained_rf(X: pd.DataFrame, y: pd.Series, n_classes: int = 2, seed: int = 42):
    """Return a fitted RandomForestClassifier."""
    clf = RandomForestClassifier(n_estimators=20, random_state=seed)
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Group 1 — Phase 1: CovarianceMetric removal
# ---------------------------------------------------------------------------

class TestCovarianceMetricRemoval:

    def test_covariance_metric_not_importable_from_top_level(self):
        """CovarianceMetric must not exist in the top-level smx namespace."""
        import smx
        assert not hasattr(smx, "CovarianceMetric"), (
            "CovarianceMetric should have been removed from smx.__init__"
        )

    def test_covariance_metric_not_importable_from_predicates(self):
        """CovarianceMetric must not exist in smx.predicates."""
        import smx.predicates as p
        assert not hasattr(p, "CovarianceMetric"), (
            "CovarianceMetric should have been removed from smx.predicates"
        )

    def test_covariance_metric_class_does_not_exist(self):
        """The class itself must not exist in the metrics module."""
        import smx.predicates.metrics as m
        assert not hasattr(m, "CovarianceMetric"), (
            "CovarianceMetric class should have been deleted from metrics.py"
        )

    def test_perturbation_metric_still_importable(self):
        """PerturbationMetric must remain accessible after the removal."""
        from smx import PerturbationMetric
        from smx.predicates.metrics import PerturbationMetric as PM2
        assert PerturbationMetric is PM2

    def test_base_predicate_metric_still_importable(self):
        """BasePredicateMetric must remain accessible."""
        from smx import BasePredicateMetric
        assert BasePredicateMetric is not None

    def test_smx_all_has_no_covariance(self):
        """'CovarianceMetric' must not appear in smx.__all__."""
        import smx
        assert "CovarianceMetric" not in smx.__all__


# ---------------------------------------------------------------------------
# Group 2 — Phase 2 / 2b: y_pred_cal removal and y_class_labels interface
# ---------------------------------------------------------------------------

class TestYPredCalRemoval:

    def test_smx_fit_does_not_accept_y_pred_cal_keyword(self):
        """Calling fit() with y_pred_cal as keyword must raise TypeError."""
        from smx import SMX
        X = make_spectral_df(60, 30)
        clf = make_trained_rf(X, pd.Series(np.tile([0, 1], 30)))
        smx = SMX(
            spectral_cuts=make_cuts(30, 3),
            quantiles=[0.25, 0.75],
            n_repetitions=1,
            n_bags=3,
            estimator=clf,
        )
        with pytest.raises(TypeError):
            smx.fit(X, y_pred_cal=clf.predict(X))  # old keyword — must fail

    def test_smx_fit_accepts_y_class_labels(self):
        """fit() must accept y_class_labels as the second positional argument."""
        from smx import SMX
        n = 80
        X = make_spectral_df(n, 40)
        y = pd.Series(np.tile([0, 1], n // 2))
        clf = make_trained_rf(X, y)
        smx = SMX(
            spectral_cuts=make_cuts(40, 4),
            quantiles=[0.25, 0.75],
            n_repetitions=2,
            n_bags=5,
            estimator=clf,
        )
        result = smx.fit(X, clf.predict(X))
        assert result is smx  # fit() must return self

    def test_smx_fit_raises_on_length_mismatch(self):
        """fit() must raise ValueError when y_class_labels has wrong length."""
        from smx import SMX
        X = make_spectral_df(60, 20)
        y = pd.Series([0, 1] * 30)
        clf = make_trained_rf(X, y)
        smx = SMX(
            spectral_cuts=make_cuts(20, 2),
            quantiles=[0.5],
            n_repetitions=1,
            n_bags=3,
            estimator=clf,
        )
        with pytest.raises(ValueError, match="length"):
            smx.fit(X, y.iloc[:30])  # wrong length

    def test_smx_fit_raises_on_single_class(self):
        """fit() must raise ValueError when all labels belong to one class."""
        from smx import SMX
        X = make_spectral_df(60, 20)
        y_single = pd.Series(["A"] * 60)
        clf = DummyClassifier(strategy="constant", constant="A")
        clf.fit(X, pd.Series(["A"] * 60))
        smx = SMX(
            spectral_cuts=make_cuts(20, 2),
            quantiles=[0.5],
            n_repetitions=1,
            n_bags=3,
            estimator=clf,
        )
        with pytest.raises(ValueError, match="2 distinct"):
            smx.fit(X, y_single)

    def test_smx_init_raises_without_estimator(self):
        """SMX.__init__() must raise ValueError when estimator is None."""
        from smx import SMX
        with pytest.raises(ValueError, match="estimator is required"):
            SMX(
                spectral_cuts=make_cuts(20, 2),
                quantiles=[0.25],
                estimator=None,
            )

    def test_smx_init_no_metric_parameter(self):
        """SMX.__init__() must not accept a 'metric' keyword argument."""
        from smx import SMX
        X = make_spectral_df(40, 20)
        clf = make_trained_rf(X, pd.Series([0, 1] * 20))
        with pytest.raises(TypeError):
            SMX(
                spectral_cuts=make_cuts(20, 2),
                quantiles=[0.25],
                estimator=clf,
                metric="perturbation",  # removed parameter — must fail
            )

    def test_smx_init_no_class_threshold_parameter(self):
        """SMX.__init__() must not accept a 'class_threshold' keyword argument."""
        from smx import SMX
        X = make_spectral_df(40, 20)
        clf = make_trained_rf(X, pd.Series([0, 1] * 20))
        with pytest.raises(TypeError):
            SMX(
                spectral_cuts=make_cuts(20, 2),
                quantiles=[0.25],
                estimator=clf,
                class_threshold=0.5,  # removed parameter — must fail
            )

    def test_smx_init_no_covariance_threshold_parameter(self):
        """SMX.__init__() must not accept a 'covariance_threshold' parameter."""
        from smx import SMX
        X = make_spectral_df(40, 20)
        clf = make_trained_rf(X, pd.Series([0, 1] * 20))
        with pytest.raises(TypeError):
            SMX(
                spectral_cuts=make_cuts(20, 2),
                quantiles=[0.25],
                estimator=clf,
                covariance_threshold=0.01,  # removed parameter — must fail
            )

    def test_bagger_run_does_not_accept_y_predicted_numeric(self):
        """PredicateBagger.run() must not accept y_predicted_numeric."""
        from smx.predicates.bagging import PredicateBagger
        from smx.predicates.generation import PredicateGenerator

        X = make_spectral_df(50, 10)
        gen = PredicateGenerator(quantiles=[0.5])
        gen.fit(X)
        bagger = PredicateBagger(random_seed=0, n_bags=2)

        with pytest.raises(TypeError):
            bagger.run(X, pd.Series(np.zeros(50)), gen.predicates_df_)

    def test_bagger_bags_have_no_predicted_y_column(self):
        """Each bag DataFrame must not contain a 'Predicted_Y' column."""
        from smx.predicates.bagging import PredicateBagger
        from smx.predicates.generation import PredicateGenerator

        X = make_spectral_df(80, 12)
        gen = PredicateGenerator(quantiles=[0.25, 0.75])
        gen.fit(X)
        bagger = PredicateBagger(random_seed=0, n_bags=3, n_samples_fraction=0.8)
        bags = bagger.run(X, gen.predicates_df_)

        for bag_name, pred_dict in bags.items():
            for rule, df in pred_dict.items():
                assert "Predicted_Y" not in df.columns, (
                    f"Bag '{bag_name}', predicate '{rule}' still has 'Predicted_Y' column."
                )

    def test_bagger_bags_have_required_columns(self):
        """Each bag DataFrame must contain 'Zone_Sum' and 'Sample_Index'."""
        from smx.predicates.bagging import PredicateBagger
        from smx.predicates.generation import PredicateGenerator

        X = make_spectral_df(80, 12)
        gen = PredicateGenerator(quantiles=[0.5])
        gen.fit(X)
        bagger = PredicateBagger(random_seed=0, n_bags=3)
        bags = bagger.run(X, gen.predicates_df_)

        for pred_dict in bags.values():
            for df in pred_dict.values():
                assert "Zone_Sum" in df.columns
                assert "Sample_Index" in df.columns


# ---------------------------------------------------------------------------
# Group 3 — Phase 3: Dynamic terminal nodes
# ---------------------------------------------------------------------------

class TestDynamicTerminalNodes:

    def test_builder_creates_correct_number_of_terminals_binary(self):
        """PredicateGraphBuilder with class_labels=['0','1'] creates 2 terminals."""
        from smx.graph.builder import PredicateGraphBuilder

        builder = PredicateGraphBuilder(class_labels=["0", "1"])
        graph = builder.build({}, {})

        terminals = [
            n for n, a in graph.nodes(data=True)
            if a.get("node_type") == "terminal"
        ]
        assert len(terminals) == 2

    def test_builder_creates_correct_number_of_terminals_multiclass(self):
        """PredicateGraphBuilder with 5 class_labels creates 5 terminals."""
        from smx.graph.builder import PredicateGraphBuilder

        labels = ["A", "B", "C", "D", "E"]
        builder = PredicateGraphBuilder(class_labels=labels)
        graph = builder.build({}, {})

        terminals = {
            n for n, a in graph.nodes(data=True)
            if a.get("node_type") == "terminal"
        }
        expected = {f"Class_{lbl}" for lbl in labels}
        assert terminals == expected

    def test_builder_terminal_names_match_class_labels(self):
        """Terminal node names must be 'Class_<label>' for each label."""
        from smx.graph.builder import PredicateGraphBuilder

        labels = ["cat", "dog", "bird"]
        builder = PredicateGraphBuilder(class_labels=labels)
        graph = builder.build({}, {})

        for lbl in labels:
            node_name = f"Class_{lbl}"
            assert graph.has_node(node_name), f"Expected terminal '{node_name}' not found."
            assert graph.nodes[node_name]["node_type"] == "terminal"
            assert graph.nodes[node_name]["class_label"] == lbl

    def test_builder_legacy_fallback_without_class_labels(self):
        """When class_labels=None, builder must create Class_A and Class_B."""
        from smx.graph.builder import PredicateGraphBuilder

        builder = PredicateGraphBuilder(class_labels=None)
        graph = builder.build({}, {})

        terminals = {
            n for n, a in graph.nodes(data=True)
            if a.get("node_type") == "terminal"
        }
        assert terminals == {"Class_A", "Class_B"}

    def test_builder_rejects_metric_column_kwarg(self):
        """build() must not accept 'metric_column' as a keyword argument."""
        from smx.graph.builder import PredicateGraphBuilder

        builder = PredicateGraphBuilder(class_labels=["0", "1"])
        with pytest.raises(TypeError):
            builder.build({}, {}, metric_column="Covariance")

    def test_last_predicate_points_to_majority_class_terminal(self):
        """The last predicate of a bag must connect to the majority-class terminal."""
        from smx.graph.builder import PredicateGraphBuilder
        from smx.predicates.generation import PredicateGenerator

        # Build a simple scenario: 20 samples, class "B" is majority in the bag
        n = 20
        class_labels = ["A", "B"]
        X = make_spectral_df(n, 10)
        gen = PredicateGenerator(quantiles=[0.5])
        gen.fit(X)

        first_rule = gen.predicates_df_["rule"].iloc[0]
        bags = {
            "Bag_1": {
                first_rule: pd.DataFrame({
                    "Zone_Sum": np.zeros(n),
                    "Sample_Index": np.arange(n),
                    "Class_Predicted": ["A"] * 4 + ["B"] * 16,  # B is majority
                })
            }
        }
        rankings = {
            "Bag_1": pd.DataFrame({
                "Predicate": [first_rule],
                "Perturbation": [0.5],
            })
        }

        builder = PredicateGraphBuilder(class_labels=class_labels)
        graph = builder.build(bags, rankings)

        # The only predicate node should have an edge to Class_B
        assert graph.has_edge(first_rule, "Class_B"), (
            "Expected edge from last predicate to 'Class_B' (majority class)."
        )
        assert not graph.has_edge(first_rule, "Class_A"), (
            "Must not have edge to non-majority 'Class_A'."
        )

    def test_integer_class_labels_are_cast_to_string(self):
        """Integer class labels (0, 1, 2) must be converted to 'Class_0', etc."""
        from smx.graph.builder import PredicateGraphBuilder

        builder = PredicateGraphBuilder(class_labels=[0, 1, 2])
        graph = builder.build({}, {})

        terminals = {
            n for n, a in graph.nodes(data=True)
            if a.get("node_type") == "terminal"
        }
        assert "Class_0" in terminals
        assert "Class_1" in terminals
        assert "Class_2" in terminals


# ---------------------------------------------------------------------------
# Group 4 — Phase 4: Pipeline integration (end-to-end)
# ---------------------------------------------------------------------------

class TestPipelineIntegration:

    def _make_smx(self, clf, n_features=40, n_zones=4):
        from smx import SMX
        return SMX(
            spectral_cuts=make_cuts(n_features, n_zones),
            quantiles=[0.25, 0.75],
            n_repetitions=2,
            n_bags=5,
            n_samples_fraction=0.8,
            estimator=clf,
            perturbation_metric="probability_shift",
            var_exp=False,
        )

    def test_end_to_end_binary(self):
        """Full pipeline must run without errors on a binary classification problem."""
        n, p = 100, 40
        X = make_spectral_df(n, p)
        y = pd.Series(np.tile([0, 1], n // 2))
        clf = make_trained_rf(X, y)

        smx = self._make_smx(clf, p)
        smx.fit(X, clf.predict(X))

        assert smx.lrc_ is not None
        assert not smx.lrc_.empty
        assert smx.lrc_unique_ is not None
        assert len(smx.valid_seeds_) > 0

    def test_end_to_end_three_classes(self):
        """Full pipeline must run without errors on a 3-class problem."""
        n, p = 120, 40
        X = make_spectral_df(n, p)
        y = pd.Series(np.tile([0, 1, 2], n // 3))
        clf = make_trained_rf(X, y, n_classes=3)

        smx = self._make_smx(clf, p)
        smx.fit(X, clf.predict(X))

        assert smx.lrc_ is not None
        assert not smx.lrc_.empty

    def test_end_to_end_five_classes(self):
        """Full pipeline must run without errors on a 5-class problem."""
        n, p = 200, 50
        X = make_spectral_df(n, p)
        y = pd.Series(np.tile([0, 1, 2, 3, 4], n // 5))
        clf = make_trained_rf(X, y, n_classes=5)

        smx = self._make_smx(clf, p, n_zones=5)
        smx.fit(X, clf.predict(X))

        assert smx.lrc_ is not None

    def test_graph_has_k_terminals_after_fit_binary(self):
        """After fitting on binary data, the graph must have exactly 2 terminals."""
        n, p = 80, 30
        X = make_spectral_df(n, p)
        y = pd.Series(np.tile([0, 1], n // 2))
        clf = make_trained_rf(X, y)

        smx = self._make_smx(clf, p, n_zones=3)
        smx.fit(X, clf.predict(X))

        # Check terminals in at least one seed graph
        for graph in smx.graphs_by_seed_.values():
            terminals = [
                n for n, a in graph.nodes(data=True)
                if a.get("node_type") == "terminal"
            ]
            assert len(terminals) == 2, (
                f"Expected 2 terminals for binary problem, got {len(terminals)}"
            )

    def test_graph_has_k_terminals_after_fit_multiclass(self):
        """After fitting on 3-class data, each seed graph must have exactly 3 terminals."""
        n, p = 120, 30
        X = make_spectral_df(n, p)
        y = pd.Series(np.tile([0, 1, 2], n // 3))
        clf = make_trained_rf(X, y, n_classes=3)

        smx = self._make_smx(clf, p)
        smx.fit(X, clf.predict(X))

        for graph in smx.graphs_by_seed_.values():
            terminals = [
                nd for nd, a in graph.nodes(data=True)
                if a.get("node_type") == "terminal"
            ]
            assert len(terminals) == 3, (
                f"Expected 3 terminals for 3-class problem, got {len(terminals)}"
            )

    def test_lrc_zones_match_spectral_cuts(self):
        """All zone names in the LRC output must correspond to defined spectral cuts."""
        n, p = 100, 40
        X = make_spectral_df(n, p)
        y = pd.Series(np.tile([0, 1], n // 2))
        clf = make_trained_rf(X, y)

        n_zones = 4
        cuts = make_cuts(p, n_zones)
        zone_names = {name for name, _, _ in cuts}

        from smx import SMX
        smx = SMX(
            spectral_cuts=cuts,
            quantiles=[0.5],
            n_repetitions=2,
            n_bags=4,
            estimator=clf,
            var_exp=False,
        )
        smx.fit(X, clf.predict(X))

        lrc_zones = set(
            smx.lrc_unique_["Zone"].dropna().tolist()
        )
        assert lrc_zones.issubset(zone_names), (
            f"Unexpected zone names in LRC output: {lrc_zones - zone_names}"
        )

    def test_fit_with_string_class_labels(self):
        """Pipeline must handle string class labels ('cat', 'dog', 'bird')."""
        n, p = 90, 30
        X = make_spectral_df(n, p)
        y = pd.Series(np.tile(["cat", "dog", "bird"], n // 3))
        clf = make_trained_rf(X, y, n_classes=3)

        from smx import SMX
        smx = SMX(
            spectral_cuts=make_cuts(p, 3),
            quantiles=[0.5],
            n_repetitions=2,
            n_bags=4,
            estimator=clf,
            var_exp=False,
        )
        y_pred = pd.Series(clf.predict(X))
        smx.fit(X, y_pred)

        assert smx.lrc_ is not None

    def test_fit_idempotent(self):
        """Calling fit() twice must overwrite previous results, not accumulate."""
        n, p = 80, 20
        X = make_spectral_df(n, p)
        y = pd.Series(np.tile([0, 1], n // 2))
        clf = make_trained_rf(X, y)

        from smx import SMX
        smx = SMX(
            spectral_cuts=make_cuts(p, 2),
            quantiles=[0.5],
            n_repetitions=1,
            n_bags=3,
            estimator=clf,
            var_exp=False,
        )
        smx.fit(X, clf.predict(X))
        first_result = smx.lrc_.copy()

        smx.fit(X, clf.predict(X))  # second call
        second_result = smx.lrc_.copy()

        pd.testing.assert_frame_equal(first_result, second_result)

    def test_bag_class_predicted_matches_y_class_labels(self):
        """Class_Predicted values in bags must be a subset of y_class_labels values."""
        from smx.predicates.bagging import PredicateBagger
        from smx.predicates.generation import PredicateGenerator

        n, p = 80, 20
        X = make_spectral_df(n, p)
        y = pd.Series(np.tile([0, 1, 2], n // 3 + 1)[:n]).astype(str)

        gen = PredicateGenerator(quantiles=[0.5])
        gen.fit(X)
        bagger = PredicateBagger(random_seed=0, n_bags=3)
        bags = bagger.run(X, gen.predicates_df_)

        # Simulate the pipeline annotation step
        for pred_dict in bags.values():
            for df_info in pred_dict.values():
                sample_indices = df_info["Sample_Index"].values
                df_info["Class_Predicted"] = y.iloc[sample_indices].values

        valid_labels = set(y.unique())
        for pred_dict in bags.values():
            for df_info in pred_dict.values():
                found = set(df_info["Class_Predicted"].tolist())
                assert found.issubset(valid_labels), (
                    f"Class_Predicted contains labels not in y: {found - valid_labels}"
                )


# ---------------------------------------------------------------------------
# Group 5 — Phase 5: Plotting colour utilities
# ---------------------------------------------------------------------------

class TestColorUtilities:

    def test_get_class_color_map_binary(self):
        """get_class_color_map must return 2 entries for 2 labels."""
        from smx.plotting._colors import get_class_color_map

        cmap = get_class_color_map(["0", "1"])
        assert len(cmap) == 2
        assert "0" in cmap
        assert "1" in cmap

    def test_get_class_color_map_five_classes(self):
        """get_class_color_map must return 5 distinct entries for 5 labels."""
        from smx.plotting._colors import get_class_color_map

        labels = ["A", "B", "C", "D", "E"]
        cmap = get_class_color_map(labels)

        assert set(cmap.keys()) == set(labels)
        # All colours must be distinct
        assert len(set(cmap.values())) == 5

    def test_get_class_color_map_custom_override(self):
        """Custom colour overrides must take precedence over palette assignment."""
        from smx.plotting._colors import get_class_color_map

        custom = {"A": "#aabbcc"}
        cmap = get_class_color_map(["A", "B"], custom_colors=custom)

        assert cmap["A"] == "#aabbcc"
        # B must still get an auto-assigned colour
        assert cmap["B"] != "#aabbcc"

    def test_get_class_color_map_hex_format(self):
        """All returned colours must be valid 7-character hex strings."""
        from smx.plotting._colors import get_class_color_map

        labels = [str(i) for i in range(12)]
        cmap = get_class_color_map(labels)

        for label, color in cmap.items():
            assert color.startswith("#"), f"Color for '{label}' does not start with '#': {color}"
            assert len(color) == 7, f"Color for '{label}' is not 7 chars: {color}"

    def test_get_class_color_map_cycles_beyond_palette(self):
        """When K exceeds palette length, colours cycle rather than error."""
        from smx.plotting._colors import get_class_color_map

        labels = [str(i) for i in range(25)]  # palette has 20 entries
        cmap = get_class_color_map(labels)

        assert len(cmap) == 25  # must not raise and must return all 25

    def test_resolve_color_returns_fallback_for_unknown_label(self):
        """resolve_color must return a fallback hex for a label not in the list."""
        from smx.plotting._colors import resolve_color

        color = resolve_color("unknown_label", class_labels=["A", "B"])
        assert color.startswith("#")
        assert len(color) == 7

    def test_get_class_color_map_importable_from_smx_plotting(self):
        """get_class_color_map must be importable from the smx.plotting package."""
        from smx.plotting import get_class_color_map
        assert callable(get_class_color_map)

    def test_theme_fallback_palette_has_at_least_12_entries(self):
        """SMXTheme.fallback_palette must contain at least 12 entries."""
        from smx.plotting.theme import SMXTheme
        theme = SMXTheme()
        assert len(theme.fallback_palette) >= 12, (
            f"Expected >= 12 entries in fallback_palette, got {len(theme.fallback_palette)}"
        )


# ---------------------------------------------------------------------------
# Group 6 — Regression / backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_builder_default_class_labels_gives_binary_terminals(self):
        """Calling PredicateGraphBuilder() without class_labels defaults to A/B."""
        from smx.graph.builder import PredicateGraphBuilder

        builder = PredicateGraphBuilder()  # class_labels defaults to None
        graph = builder.build({}, {})

        terminals = {
            n for n, a in graph.nodes(data=True)
            if a.get("node_type") == "terminal"
        }
        assert "Class_A" in terminals
        assert "Class_B" in terminals

    def test_perturbation_metric_probability_shift_binary(self):
        """probability_shift must return a scalar in [0,1] for a binary classifier."""
        from smx.predicates.metrics import PerturbationMetric
        from smx.predicates.generation import PredicateGenerator
        from smx.predicates.bagging import PredicateBagger

        n, p = 60, 10
        X = make_spectral_df(n, p)
        y = pd.Series(np.tile([0, 1], n // 2))
        clf = make_trained_rf(X, y)

        gen = PredicateGenerator(quantiles=[0.5])
        gen.fit(X)
        bagger = PredicateBagger(random_seed=0, n_bags=2)
        bags = bagger.run(X, gen.predicates_df_)

        # Annotate with class labels (simulating what pipeline does)
        y_labels = pd.Series(clf.predict(X)).astype(str)
        for pred_dict in bags.values():
            for df_info in pred_dict.values():
                df_info["Class_Predicted"] = y_labels.iloc[
                    df_info["Sample_Index"].values
                ].values

        metric = PerturbationMetric(
            estimator=clf,
            Xcalclass_prep=X,
            predicates_df=gen.predicates_df_,
            spectral_cuts=make_cuts(p, 2),
            metric="probability_shift",
        )
        rankings = metric.compute(bags)

        for bag_name, df in rankings.items():
            if bag_name.startswith("__"):
                continue
            for val in df["Perturbation"]:
                assert 0.0 <= val <= 1.0, (
                    f"probability_shift value {val} outside [0,1] in {bag_name}"
                )

    def test_perturbation_metric_probability_shift_multiclass(self):
        """probability_shift must return a scalar in [0,1] for a 3-class classifier."""
        from smx.predicates.metrics import PerturbationMetric
        from smx.predicates.generation import PredicateGenerator
        from smx.predicates.bagging import PredicateBagger

        n, p = 90, 12
        X = make_spectral_df(n, p)
        y = pd.Series(np.tile([0, 1, 2], n // 3))
        clf = make_trained_rf(X, y, n_classes=3)

        gen = PredicateGenerator(quantiles=[0.5])
        gen.fit(X)
        bagger = PredicateBagger(random_seed=0, n_bags=2)
        bags = bagger.run(X, gen.predicates_df_)

        y_labels = pd.Series(clf.predict(X)).astype(str)
        for pred_dict in bags.values():
            for df_info in pred_dict.values():
                df_info["Class_Predicted"] = y_labels.iloc[
                    df_info["Sample_Index"].values
                ].values

        metric = PerturbationMetric(
            estimator=clf,
            Xcalclass_prep=X,
            predicates_df=gen.predicates_df_,
            spectral_cuts=make_cuts(p, 3),
            metric="probability_shift",
        )
        rankings = metric.compute(bags)

        for bag_name, df in rankings.items():
            if bag_name.startswith("__"):
                continue
            for val in df["Perturbation"]:
                assert 0.0 <= val <= 1.0, (
                    f"probability_shift value {val} outside [0,1] for 3-class in {bag_name}"
                )

    def test_probability_shift_tvd_formula_invariance(self):
        """TVD formula must return identical results for K=2 and equivalent K=2 binary."""
        # Verify that dividing by 2.0 is correct: for binary, TVD = |p1_a - p1_b|
        prob_orig = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
        prob_pert = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        tvd_formula = np.mean(np.sum(np.abs(prob_orig - prob_pert), axis=1) / 2.0)
        scalar_formula = np.mean(np.abs(prob_orig[:, 1] - prob_pert[:, 1]))

        assert abs(tvd_formula - scalar_formula) < 1e-12, (
            "TVD formula must be equivalent to scalar difference for K=2"
        )

    def test_probability_shift_tvd_bounded_for_k3(self):
        """TVD result must be in [0, 1] for a K=3 case."""
        # Extreme: all probability shifts to a different class
        prob_orig = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        prob_pert = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        tvd = np.mean(np.sum(np.abs(prob_orig - prob_pert), axis=1) / 2.0)
        assert tvd == pytest.approx(1.0), f"Max TVD for K=3 should be 1.0, got {tvd}"

        # Minimal shift
        prob_same = prob_orig.copy()
        tvd_zero = np.mean(np.sum(np.abs(prob_orig - prob_same), axis=1) / 2.0)
        assert tvd_zero == pytest.approx(0.0)
