"""Class colour utilities (migrated unchanged from the former test_multiclass.py)."""


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
