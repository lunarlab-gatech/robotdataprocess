import matplotlib
matplotlib.use('Agg')

import dataclasses
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import os
import pandas as pd
from pathlib import Path
import tempfile
import unittest
import unittest.mock

from robotdataprocess.utils.TablePlotter import TablePlotter


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestFormattedTextSegment(unittest.TestCase):
    """Test the FormattedTextSegment dataclass."""

    def test_defaults(self):
        seg = TablePlotter.FormattedTextSegment("42")
        self.assertEqual(seg.text, "42")
        self.assertFalse(seg.bold)
        self.assertFalse(seg.underline)
        self.assertIsNone(seg.color)

    def test_all_fields_set(self):
        seg = TablePlotter.FormattedTextSegment("42", bold=True, underline=True, color='#1A3055')
        self.assertEqual(seg.text, "42")
        self.assertTrue(seg.bold)
        self.assertTrue(seg.underline)
        self.assertEqual(seg.color, '#1A3055')

    def test_equality(self):
        a = TablePlotter.FormattedTextSegment("42", bold=True)
        b = TablePlotter.FormattedTextSegment("42", bold=True)
        c = TablePlotter.FormattedTextSegment("42", bold=False)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_frozen(self):
        seg = TablePlotter.FormattedTextSegment("42")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            seg.text = "43"


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestTableStyle(unittest.TestCase):
    """Test TableStyleName / TableStyle / get_table_style."""

    def test_georgia_tech_style(self):
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        self.assertEqual(style.HeaderColor, '#B5A060')
        self.assertEqual(style.HeaderTextColor, '#FFFFFF')
        self.assertEqual(style.TextColor, '#1A3055')
        self.assertEqual(style.TextFailureColor, '#CC2222')
        self.assertEqual(style.RowColors, ('#EDEADE', '#D8D3C3'))

    def test_byu_style(self):
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(style.HeaderColor, '#2F61B3')
        self.assertEqual(style.HeaderTextColor, '#FFFFFF')
        self.assertEqual(style.TextColor, '#002E5D')
        self.assertEqual(style.TextFailureColor, '#9E2A2B')
        self.assertEqual(style.RowColors, ("#DBDEE8", "#C9CEDE"))

    def test_styles_are_distinct(self):
        gt = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        byu = TablePlotter.get_table_style(TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertNotEqual(gt, byu)

    def test_table_style_is_frozen(self):
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            style.HeaderColor = '#000000'

    def test_unsupported_style_raises(self):
        # Fake enum-like sentinel that isn't a real TableStyleName member.
        with self.assertRaises(ValueError):
            TablePlotter.get_table_style(None)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestRankDataInSeries(unittest.TestCase):
    """Test the internal _rank_data_in_Series ranking helper."""

    def test_higher_is_better_default(self):
        data = pd.Series({'a': 3.0, 'b': 1.0, 'c': 2.0})
        groups = TablePlotter._rank_data_in_Series(data)
        self.assertEqual(groups, [{'a'}, {'c'}, {'b'}])

    def test_lower_is_better(self):
        data = pd.Series({'a': 3.0, 'b': 1.0, 'c': 2.0})
        groups = TablePlotter._rank_data_in_Series(data, higher_is_better=False)
        self.assertEqual(groups, [{'b'}, {'c'}, {'a'}])

    def test_ties_grouped_together(self):
        data = pd.Series({'a': 5.0, 'b': 5.0, 'c': 1.0})
        groups = TablePlotter._rank_data_in_Series(data)
        self.assertEqual(groups, [{'a', 'b'}, {'c'}])

    def test_nan_excluded_from_all_groups(self):
        data = pd.Series({'a': 5.0, 'b': float('nan'), 'c': 1.0})
        groups = TablePlotter._rank_data_in_Series(data)
        for group in groups:
            self.assertNotIn('b', group)
        self.assertEqual(groups, [{'a'}, {'c'}])

    def test_all_nan_returns_no_groups(self):
        data = pd.Series({'a': float('nan'), 'b': float('nan')})
        groups = TablePlotter._rank_data_in_Series(data)
        self.assertEqual(groups, [])

    def test_single_value(self):
        data = pd.Series({'a': 1.0})
        groups = TablePlotter._rank_data_in_Series(data)
        self.assertEqual(groups, [{'a'}])


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestConvertNumbersToTextSegments(unittest.TestCase):
    """Test convert_numbers_to_TextSegments."""

    def test_returns_series_indexed_like_input(self):
        data = pd.Series({'a': 3.0, 'b': 1.0, 'c': 2.0})
        result = TablePlotter.convert_numbers_to_TextSegments(data)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(list(result.index), ['a', 'b', 'c'])
        for segs in result:
            self.assertEqual(len(segs), 1)
            self.assertIsInstance(segs[0], TablePlotter.FormattedTextSegment)

    def test_best_value_is_bold_second_is_underlined(self):
        data = pd.Series({'a': 3.0, 'b': 1.0, 'c': 2.0})
        result = TablePlotter.convert_numbers_to_TextSegments(data)
        self.assertTrue(result['a'][0].bold)
        self.assertFalse(result['a'][0].underline)
        self.assertTrue(result['c'][0].underline)
        self.assertFalse(result['c'][0].bold)
        self.assertFalse(result['b'][0].bold)
        self.assertFalse(result['b'][0].underline)

    def test_higher_is_better_false_flips_ranking(self):
        data = pd.Series({'a': 3.0, 'b': 1.0, 'c': 2.0})
        result = TablePlotter.convert_numbers_to_TextSegments(data, higher_is_better=False)
        self.assertTrue(result['b'][0].bold)
        self.assertTrue(result['c'][0].underline)
        self.assertFalse(result['a'][0].bold)
        self.assertFalse(result['a'][0].underline)

    def test_ties_all_bolded(self):
        data = pd.Series({'a': 5.0, 'b': 5.0, 'c': 1.0})
        result = TablePlotter.convert_numbers_to_TextSegments(data)
        self.assertTrue(result['a'][0].bold)
        self.assertTrue(result['b'][0].bold)
        self.assertFalse(result['c'][0].bold)
        self.assertTrue(result['c'][0].underline)  # sole remaining value -> 2nd-place group

    def test_emphasize_rankings_false_disables_bold_and_underline(self):
        data = pd.Series({'a': 3.0, 'b': 1.0, 'c': 2.0})
        result = TablePlotter.convert_numbers_to_TextSegments(data, emphasize_rankings=False)
        for segs in result:
            self.assertFalse(segs[0].bold)
            self.assertFalse(segs[0].underline)

    def test_default_fmt_is_str(self):
        data = pd.Series({'a': 3.0})
        result = TablePlotter.convert_numbers_to_TextSegments(data)
        self.assertEqual(result['a'][0].text, str(3.0))

    def test_custom_fmt_applied(self):
        data = pd.Series({'a': 3.0, 'b': float('nan')})
        result = TablePlotter.convert_numbers_to_TextSegments(data, fmt=TablePlotter.fmt_fixed(1, suffix='%'))
        self.assertEqual(result['a'][0].text, "3.0%")
        self.assertEqual(result['b'][0].text, "---")

    def test_custom_color_fn_applied(self):
        data = pd.Series({'a': 3.0, 'b': 1.0})
        result = TablePlotter.convert_numbers_to_TextSegments(data, color_fn=lambda v: 'green' if v > 2 else 'red')
        self.assertEqual(result['a'][0].color, 'green')
        self.assertEqual(result['b'][0].color, 'red')

    def test_default_color_fn_flags_nan_as_failure(self):
        data = pd.Series({'a': 3.0, 'b': float('nan')})
        result = TablePlotter.convert_numbers_to_TextSegments(data)
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        self.assertEqual(result['b'][0].color, style.TextFailureColor)
        self.assertEqual(result['a'][0].color, style.TextColor)

    def test_nan_never_bold_or_underlined_even_if_tied(self):
        # NaN values are dropped before ranking, so even a "tie" among all-NaN
        # entries never produces bold/underline.
        data = pd.Series({'a': float('nan'), 'b': float('nan'), 'c': 1.0})
        result = TablePlotter.convert_numbers_to_TextSegments(data)
        self.assertFalse(result['a'][0].bold)
        self.assertFalse(result['a'][0].underline)
        self.assertFalse(result['b'][0].bold)
        self.assertFalse(result['b'][0].underline)
        self.assertTrue(result['c'][0].bold)

    def test_single_value_series_is_bold(self):
        data = pd.Series({'a': 1.0})
        result = TablePlotter.convert_numbers_to_TextSegments(data)
        self.assertTrue(result['a'][0].bold)
        self.assertFalse(result['a'][0].underline)

    def test_empty_series(self):
        data = pd.Series({}, dtype=float)
        result = TablePlotter.convert_numbers_to_TextSegments(data)
        self.assertEqual(len(result), 0)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestMergeSeries(unittest.TestCase):
    """Test merge_Series."""

    def test_merges_two_series_with_default_separator(self):
        a = pd.Series({'x': [TablePlotter.FormattedTextSegment("1")]})
        b = pd.Series({'x': [TablePlotter.FormattedTextSegment("2")]})
        merged = TablePlotter.merge_Series(a, b)
        self.assertIsInstance(merged, pd.Series)
        texts = [seg.text for seg in merged['x']]
        self.assertEqual(texts, ["1", "/", "2"])

    def test_custom_separator(self):
        a = pd.Series({'x': [TablePlotter.FormattedTextSegment("1")]})
        b = pd.Series({'x': [TablePlotter.FormattedTextSegment("2")]})
        merged = TablePlotter.merge_Series(a, b, separator='-')
        texts = [seg.text for seg in merged['x']]
        self.assertEqual(texts, ["1", "-", "2"])

    def test_merges_three_series(self):
        a = pd.Series({'x': [TablePlotter.FormattedTextSegment("1")]})
        b = pd.Series({'x': [TablePlotter.FormattedTextSegment("2")]})
        c = pd.Series({'x': [TablePlotter.FormattedTextSegment("3")]})
        merged = TablePlotter.merge_Series(a, b, c)
        texts = [seg.text for seg in merged['x']]
        self.assertEqual(texts, ["1", "/", "2", "/", "3"])

    def test_single_series_is_passthrough_with_no_separator(self):
        a = pd.Series({'x': [TablePlotter.FormattedTextSegment("1")]})
        merged = TablePlotter.merge_Series(a)
        texts = [seg.text for seg in merged['x']]
        self.assertEqual(texts, ["1"])

    def test_preserves_all_keys(self):
        a = pd.Series({'x': [TablePlotter.FormattedTextSegment("1")], 'y': [TablePlotter.FormattedTextSegment("3")]})
        b = pd.Series({'x': [TablePlotter.FormattedTextSegment("2")], 'y': [TablePlotter.FormattedTextSegment("4")]})
        merged = TablePlotter.merge_Series(a, b)
        self.assertEqual(set(merged.index), {'x', 'y'})

    def test_separator_color_defaults_to_georgia_tech_text_color(self):
        a = pd.Series({'x': [TablePlotter.FormattedTextSegment("1")]})
        b = pd.Series({'x': [TablePlotter.FormattedTextSegment("2")]})
        merged = TablePlotter.merge_Series(a, b)
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        self.assertEqual(merged['x'][1].color, style.TextColor)

    def test_separator_color_uses_given_style(self):
        a = pd.Series({'x': [TablePlotter.FormattedTextSegment("1")]})
        b = pd.Series({'x': [TablePlotter.FormattedTextSegment("2")]})
        merged = TablePlotter.merge_Series(a, b, style=TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(merged['x'][1].color, style.TextColor)

    def test_merges_output_of_convert_numbers_to_TextSegments(self):
        a = TablePlotter.convert_numbers_to_TextSegments(pd.Series({'r1': 5.0, 'r2': 3.0}))
        b = TablePlotter.convert_numbers_to_TextSegments(pd.Series({'r1': 50.0, 'r2': 30.0}))
        merged = TablePlotter.merge_Series(a, b)
        texts = [seg.text for seg in merged['r1']]
        self.assertEqual(texts, ["5.0", "/", "50.0"])


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestColorFunctions(unittest.TestCase):
    """Test color_fn_NAVY_RED_missing_or_equal / color_fn_NAVY_RED_missing_or_above."""

    def test_missing_or_equal_flags_nan(self):
        color_fn = TablePlotter.color_fn_NAVY_RED_missing_or_equal()
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        self.assertEqual(color_fn(float('nan')), style.TextFailureColor)

    def test_missing_or_equal_flags_equal_value(self):
        color_fn = TablePlotter.color_fn_NAVY_RED_missing_or_equal(equal_value=5.0)
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        self.assertEqual(color_fn(5.0), style.TextFailureColor)
        self.assertEqual(color_fn(0.0), style.TextColor)

    def test_missing_or_equal_default_equal_value_is_zero(self):
        color_fn = TablePlotter.color_fn_NAVY_RED_missing_or_equal()
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        self.assertEqual(color_fn(0.0), style.TextFailureColor)
        self.assertEqual(color_fn(1.0), style.TextColor)

    def test_missing_or_above_flags_nan_and_above_threshold(self):
        color_fn = TablePlotter.color_fn_NAVY_RED_missing_or_above(10.0)
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        self.assertEqual(color_fn(float('nan')), style.TextFailureColor)
        self.assertEqual(color_fn(10.1), style.TextFailureColor)
        self.assertEqual(color_fn(10.0), style.TextColor)
        self.assertEqual(color_fn(-5.0), style.TextColor)

    def test_missing_or_above_infinite_threshold_never_flags_real_values(self):
        color_fn = TablePlotter.color_fn_NAVY_RED_missing_or_above(float('inf'))
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        self.assertEqual(color_fn(1e300), style.TextColor)
        self.assertEqual(color_fn(float('nan')), style.TextFailureColor)

    def test_color_fns_respect_style_parameter(self):
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        eq_fn = TablePlotter.color_fn_NAVY_RED_missing_or_equal(style=TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        above_fn = TablePlotter.color_fn_NAVY_RED_missing_or_above(10.0, style=TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(eq_fn(1.0), style.TextColor)
        self.assertEqual(above_fn(1.0), style.TextColor)

    def test_color_fns_return_independent_closures(self):
        # Two calls with different thresholds must not share state.
        fn_low = TablePlotter.color_fn_NAVY_RED_missing_or_above(1.0)
        fn_high = TablePlotter.color_fn_NAVY_RED_missing_or_above(100.0)
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        self.assertEqual(fn_low(5.0), style.TextFailureColor)
        self.assertEqual(fn_high(5.0), style.TextColor)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestFmtFixed(unittest.TestCase):
    """Test fmt_fixed."""

    def test_default_precision(self):
        fmt = TablePlotter.fmt_fixed()
        self.assertEqual(fmt(3.14159), "3.14")

    def test_custom_precision(self):
        fmt = TablePlotter.fmt_fixed(precision=0)
        self.assertEqual(fmt(3.9), "4")

    def test_suffix_applied(self):
        fmt = TablePlotter.fmt_fixed(precision=1, suffix='%')
        self.assertEqual(fmt(50.0), "50.0%")

    def test_nan_uses_missing_str(self):
        fmt = TablePlotter.fmt_fixed(missing_str="N/A")
        self.assertEqual(fmt(float('nan')), "N/A")

    def test_default_missing_str_is_dashes(self):
        fmt = TablePlotter.fmt_fixed()
        self.assertEqual(fmt(float('nan')), "---")


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestRenderTableOntoAx(unittest.TestCase):
    """Test render_table_onto_ax."""

    def _make_df(self, values=None, title="Method"):
        values = values if values is not None else {'ColA': {'r1': 5.0, 'r2': 3.0, 'r3': 1.0}}
        styled = {col: TablePlotter.convert_numbers_to_TextSegments(pd.Series(vals))
                  for col, vals in values.items()}
        df = pd.DataFrame(styled)
        df.attrs["title"] = title
        return df

    def setUp(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 3))
        self.ax.axis('off')
        self.fig.tight_layout(pad=0.0)

    def tearDown(self):
        plt.close(self.fig)

    def test_renders_without_raising(self):
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df)

    def test_missing_title_raises_key_error(self):
        df = pd.DataFrame({'ColA': TablePlotter.convert_numbers_to_TextSegments(pd.Series({'r1': 1.0}))})
        with self.assertRaises(KeyError):
            TablePlotter.render_table_onto_ax(self.fig, self.ax, df)

    def test_column_labels_include_title_and_columns(self):
        df = self._make_df(title="MyTitle")
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df)
        tbl = [c for c in self.ax.tables][0]
        self.assertEqual(tbl[0, 0].get_text().get_text(), "MyTitle")
        self.assertEqual(tbl[0, 1].get_text().get_text(), "ColA")

    def test_row_labels_are_index_values(self):
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df)
        tbl = [c for c in self.ax.tables][0]
        self.assertEqual(tbl[1, 0].get_text().get_text(), "r1")
        self.assertEqual(tbl[2, 0].get_text().get_text(), "r2")
        self.assertEqual(tbl[3, 0].get_text().get_text(), "r3")

    def test_header_facecolor_matches_style(self):
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df)
        tbl = [c for c in self.ax.tables][0]
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        self.assertEqual(tbl[0, 0].get_facecolor(), to_rgba(style.HeaderColor))

    def test_header_facecolor_matches_given_style(self):
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df, style=TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        tbl = [c for c in self.ax.tables][0]
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(tbl[0, 0].get_facecolor(), to_rgba(style.HeaderColor))

    def test_row_facecolors_alternate(self):
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df)
        tbl = [c for c in self.ax.tables][0]
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.GEORGIA_TECH)
        self.assertEqual(tbl[1, 0].get_facecolor(), to_rgba(style.RowColors[0]))
        self.assertEqual(tbl[2, 0].get_facecolor(), to_rgba(style.RowColors[1]))
        self.assertEqual(tbl[3, 0].get_facecolor(), to_rgba(style.RowColors[0]))

    def test_header_text_color_matches_style(self):
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df, style=TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        tbl = [c for c in self.ax.tables][0]
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(tbl[0, 1].get_text().get_color(), style.HeaderTextColor)

    def test_row_label_text_color_matches_style_text_color(self):
        """Regression test: row labels (method names) must use the style's
        TextColor, not matplotlib's default text color."""
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df, style=TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        tbl = [c for c in self.ax.tables][0]
        style = TablePlotter.get_table_style(TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(tbl[1, 0].get_text().get_color(), style.TextColor)
        self.assertEqual(tbl[2, 0].get_text().get_color(), style.TextColor)
        self.assertEqual(tbl[3, 0].get_text().get_color(), style.TextColor)

    def test_data_font_size_overrides_font_size(self):
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df, font_size=11, data_font_size=20)
        tbl = [c for c in self.ax.tables][0]
        self.assertEqual(tbl[0, 1].get_text().get_fontsize(), 11)  # header keeps font_size
        self.assertEqual(tbl[1, 1].get_text().get_fontsize(), 20)  # data cell overridden

    def test_simple_cell_text_and_bold_applied(self):
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df)
        tbl = [c for c in self.ax.tables][0]
        # r1 (5.0) is the best value -> bold, non-underlined -> rendered directly in-cell.
        self.assertEqual(tbl[1, 1].get_text().get_text(), "5.0")
        self.assertEqual(tbl[1, 1].get_text().get_fontweight(), 'bold')

    def test_underlined_cell_text_cleared_and_redrawn_via_fig_text(self):
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df)
        tbl = [c for c in self.ax.tables][0]
        # r2 (3.0) is second-best -> underlined -> in-cell text wiped, redrawn as fig.text.
        self.assertEqual(tbl[2, 1].get_text().get_text(), "")
        redrawn_texts = [t.get_text() for t in self.fig.texts]
        self.assertIn("3.0", redrawn_texts)

    def test_multi_segment_cell_produces_one_fig_text_per_segment(self):
        a = TablePlotter.convert_numbers_to_TextSegments(pd.Series({'r1': 5.0}))
        b = TablePlotter.convert_numbers_to_TextSegments(pd.Series({'r1': 50.0}))
        merged = TablePlotter.merge_Series(a, b)
        df = pd.DataFrame({'ColA': merged})
        df.attrs["title"] = "Method"
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df)
        redrawn_texts = sorted(t.get_text() for t in self.fig.texts)
        self.assertEqual(redrawn_texts, ["/", "5.0", "50.0"])

    def test_heavy_divider_before_called_with_zero_based_column_indices(self):
        df = self._make_df(values={'ColA': {'r1': 1.0}, 'ColB': {'r1': 2.0}, 'ColC': {'r1': 3.0}})
        heavy_fn = unittest.mock.Mock(return_value=False)
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df, heavy_divider_before=heavy_fn)
        self.assertEqual(heavy_fn.call_args_list, [unittest.mock.call(0), unittest.mock.call(1), unittest.mock.call(2)])

    def test_heavy_divider_is_white_but_thicker(self):
        df = self._make_df(values={'ColA': {'r1': 1.0}, 'ColB': {'r1': 2.0}})
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df, heavy_divider_before=lambda _: True)
        divider_lines = [line for line in self.fig.artists if isinstance(line, mlines.Line2D)]
        self.assertTrue(divider_lines)
        for line in divider_lines:
            self.assertEqual(line.get_color(), 'white')
            self.assertEqual(line.get_linewidth(), 2.2)

    def test_non_heavy_divider_is_white_and_thin(self):
        df = self._make_df(values={'ColA': {'r1': 1.0}, 'ColB': {'r1': 2.0}})
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df, heavy_divider_before=lambda _: False)
        divider_lines = [line for line in self.fig.artists if isinstance(line, mlines.Line2D)]
        self.assertTrue(divider_lines)
        for line in divider_lines:
            self.assertEqual(line.get_color(), 'white')
            self.assertEqual(line.get_linewidth(), 1.0)

    def test_default_tbl_bbox_is_full_axes(self):
        df = self._make_df()
        # Should not raise, and should use the full [0, 0, 1, 1] axes bbox by default.
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df, tbl_bbox=None)
        tbl = [c for c in self.ax.tables][0]
        self.assertIsNotNone(tbl)

    def test_custom_tbl_bbox_accepted(self):
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df, tbl_bbox=[0, 0.3, 0.75, 0.3])

    def test_overlay_artists_use_figure_fraction_transform_not_identity(self):
        """Regression test: overlay text/lines must use fig.transFigure, not raw
        display pixels, so they stay correctly positioned after a later
        ``savefig(bbox_inches='tight')`` recrops the canvas (see bug where
        underlines/dividers rendered in the wrong place entirely)."""
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df)

        # All post-render fig.text() glyphs (multi-segment/underlined values)
        # must be anchored to fig.transFigure.
        for txt in self.fig.texts:
            self.assertIs(txt.get_transform(), self.fig.transFigure)

        # All divider/underline lines added via fig.add_artist must likewise
        # be anchored to fig.transFigure.
        for artist in self.fig.artists:
            if isinstance(artist, mlines.Line2D):
                self.assertIs(artist.get_transform(), self.fig.transFigure)

    def test_underline_survives_bbox_inches_tight_savefig(self):
        """End-to-end regression test for the tight-bbox positioning bug:
        render a table with an underlined value, save with
        bbox_inches='tight', and verify the underline line ends up directly
        beneath its text (not floating elsewhere on the page)."""
        df = self._make_df()
        TablePlotter.render_table_onto_ax(self.fig, self.ax, df)

        underline_texts = [t for t in self.fig.texts if t.get_text() == "3.0"]
        self.assertEqual(len(underline_texts), 1)
        underline_lines = [a for a in self.fig.artists
                            if isinstance(a, mlines.Line2D) and a.get_linewidth() == 0.8]
        self.assertEqual(len(underline_lines), 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "out.pdf"
            self.fig.savefig(str(save_path), bbox_inches='tight')
            self.assertTrue(save_path.exists())

        # Renderer-measured extents (in figure-fraction space, stable across
        # the later tight-bbox save) should place the underline directly
        # under its text: same x-center, underline y strictly below text y.
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        text_bb = underline_texts[0].get_window_extent(renderer=renderer)
        line_xdata, _ = underline_lines[0].get_data()
        line_bb_x0, line_bb_x1 = self.fig.transFigure.transform((line_xdata[0], 0))[0], \
            self.fig.transFigure.transform((line_xdata[1], 0))[0]
        text_cx = (text_bb.x0 + text_bb.x1) / 2
        self.assertAlmostEqual(line_bb_x0, text_bb.x0, delta=2.0)
        self.assertAlmostEqual(line_bb_x1, text_bb.x1, delta=2.0)
        self.assertTrue(line_bb_x0 < text_cx < line_bb_x1)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestPlotTablesOnPdf(unittest.TestCase):
    """Test plot_tables_on_pdf."""

    def _make_df(self, title="Method"):
        styled = TablePlotter.convert_numbers_to_TextSegments(pd.Series({'r1': 5.0, 'r2': 3.0, 'r3': 1.0}))
        df = pd.DataFrame({'ColA': styled})
        df.attrs["title"] = title
        return df

    def test_saves_single_table_pdf(self):
        df = self._make_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "out.pdf"
            TablePlotter.plot_tables_on_pdf([df], str(save_path))
            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 0)

    def test_saves_multiple_tables_pdf(self):
        df1 = self._make_df(title="Table1")
        df2 = self._make_df(title="Table2")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "out.pdf"
            TablePlotter.plot_tables_on_pdf([df1, df2], str(save_path))
            self.assertTrue(save_path.exists())

    def test_saves_as_png(self):
        df = self._make_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "out.png"
            TablePlotter.plot_tables_on_pdf([df], str(save_path))
            self.assertTrue(save_path.exists())

    def test_creates_missing_parent_directory(self):
        df = self._make_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dir" / "out.pdf"
            # plot_tables_on_pdf itself doesn't mkdir; verify caller-created
            # parent dirs are respected and no unrelated error is raised when
            # the directory already exists.
            save_path.parent.mkdir(parents=True, exist_ok=True)
            TablePlotter.plot_tables_on_pdf([df], str(save_path))
            self.assertTrue(save_path.exists())

    def test_style_forwarded_to_render_table_onto_ax(self):
        df = self._make_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "out.pdf"
            with unittest.mock.patch.object(TablePlotter, 'render_table_onto_ax',
                                             wraps=TablePlotter.render_table_onto_ax) as spy:
                TablePlotter.plot_tables_on_pdf([df], str(save_path), style=TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
                _, kwargs = spy.call_args
                self.assertEqual(kwargs['style'], TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)

    def test_default_style_is_georgia_tech(self):
        df = self._make_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "out.pdf"
            with unittest.mock.patch.object(TablePlotter, 'render_table_onto_ax',
                                             wraps=TablePlotter.render_table_onto_ax) as spy:
                TablePlotter.plot_tables_on_pdf([df], str(save_path))
                _, kwargs = spy.call_args
                self.assertEqual(kwargs['style'], TablePlotter.TableStyleName.GEORGIA_TECH)


if __name__ == '__main__':
    unittest.main()
