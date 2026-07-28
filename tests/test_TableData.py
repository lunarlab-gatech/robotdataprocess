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

from robotdataprocess.data_types.TableData import TableData
from typeguard import TypeCheckError


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestFormattedTextSegment(unittest.TestCase):
    """Test the FormattedTextSegment dataclass."""

    def test_defaults(self):
        seg = TableData.FormattedTextSegment("42")
        self.assertEqual(seg.text, "42")
        self.assertEqual(seg.styles, [])
        self.assertIsNone(seg.color)
        self.assertIsNone(seg.raw_value)

    def test_all_fields_set(self):
        seg = TableData.FormattedTextSegment(
            "42", styles=[TableData.TextStyle.BOLD, TableData.TextStyle.UNDERLINE],
            color='#1A3055', raw_value=42)
        self.assertEqual(seg.text, "42")
        self.assertIn(TableData.TextStyle.BOLD, seg.styles)
        self.assertIn(TableData.TextStyle.UNDERLINE, seg.styles)
        self.assertEqual(seg.color, '#1A3055')
        self.assertEqual(seg.raw_value, 42)

    def test_equality(self):
        a = TableData.FormattedTextSegment("42", styles=[TableData.TextStyle.BOLD])
        b = TableData.FormattedTextSegment("42", styles=[TableData.TextStyle.BOLD])
        c = TableData.FormattedTextSegment("42", styles=[])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_frozen(self):
        seg = TableData.FormattedTextSegment("42")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            seg.text = "43"


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestTableStyle(unittest.TestCase):
    """Test TableStyleName / TableStyle / get_table_style."""

    def test_georgia_tech_style(self):
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        self.assertEqual(style.HeaderColor, '#B5A060')
        self.assertEqual(style.HeaderTextColor, '#FFFFFF')
        self.assertEqual(style.TextColor, '#1A3055')
        self.assertEqual(style.TextFailureColor, '#CC2222')
        self.assertEqual(style.RowColors, ('#EDEADE', '#D8D3C3'))

    def test_byu_style(self):
        style = TableData.get_table_style(TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(style.HeaderColor, '#2F61B3')
        self.assertEqual(style.HeaderTextColor, '#FFFFFF')
        self.assertEqual(style.TextColor, '#002E5D')
        self.assertEqual(style.TextFailureColor, '#9E2A2B')
        self.assertEqual(style.RowColors, ("#DBDEE8", "#C9CEDE"))

    def test_latex_style(self):
        style = TableData.get_table_style(TableData.TableStyleName.LATEX)
        self.assertIsNone(style.HeaderColor)
        self.assertIsNone(style.HeaderTextColor)
        self.assertIsNone(style.TextColor)
        self.assertEqual(style.TextFailureColor, '#CC2222')
        self.assertIsNone(style.RowColors)

    def test_styles_are_distinct(self):
        gt = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        byu = TableData.get_table_style(TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertNotEqual(gt, byu)

    def test_table_style_is_frozen(self):
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            style.HeaderColor = '#000000'

    def test_unsupported_style_raises(self):
        # TableData is @typechecked, so a non-TableStyleName argument is
        # rejected before get_table_style's own ValueError check ever runs.
        with self.assertRaises(TypeCheckError):
            TableData.get_table_style(None)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestRankDataInSeries(unittest.TestCase):
    """Test the internal _rank_data_in_Series ranking helper (returns integer positions, not labels)."""

    def test_higher_is_better_default(self):
        data = pd.Series({'a': 3.0, 'b': 1.0, 'c': 2.0})
        groups = TableData._rank_data_in_Series(data)
        self.assertEqual(groups, [{0}, {2}, {1}])

    def test_lower_is_better(self):
        data = pd.Series({'a': 3.0, 'b': 1.0, 'c': 2.0})
        groups = TableData._rank_data_in_Series(data, higher_is_better=False)
        self.assertEqual(groups, [{1}, {2}, {0}])

    def test_ties_grouped_together(self):
        data = pd.Series({'a': 5.0, 'b': 5.0, 'c': 1.0})
        groups = TableData._rank_data_in_Series(data)
        self.assertEqual(groups, [{0, 1}, {2}])

    def test_nan_excluded_from_all_groups(self):
        data = pd.Series({'a': 5.0, 'b': float('nan'), 'c': 1.0})
        groups = TableData._rank_data_in_Series(data)
        for group in groups:
            self.assertNotIn(1, group)  # position of 'b'
        self.assertEqual(groups, [{0}, {2}])

    def test_all_nan_returns_no_groups(self):
        data = pd.Series({'a': float('nan'), 'b': float('nan')})
        groups = TableData._rank_data_in_Series(data)
        self.assertEqual(groups, [])

    def test_single_value(self):
        data = pd.Series({'a': 1.0})
        groups = TableData._rank_data_in_Series(data)
        self.assertEqual(groups, [{0}])


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestFromDataFrame(unittest.TestCase):
    """Test TableData.from_DataFrame / from_formatted_DataFrame."""

    def test_wraps_raw_values_as_single_segment(self):
        df = pd.DataFrame({'ColA': {'a': 3.0, 'b': 1.0}})
        table = TableData.from_DataFrame(df)
        self.assertEqual(table.df.at['a', 'ColA'], [TableData.FormattedTextSegment("3.0", raw_value=3.0)])

    def test_preserves_title(self):
        df = pd.DataFrame({'ColA': {'a': 3.0}})
        df.attrs["title"] = "MyTitle"
        table = TableData.from_DataFrame(df)
        self.assertEqual(table.df.attrs["title"], "MyTitle")

    def test_from_formatted_DataFrame_accepts_valid_cells(self):
        df = pd.DataFrame({'ColA': [[TableData.FormattedTextSegment("1")]]}, index=['x'])
        table = TableData.from_formatted_DataFrame(df)
        self.assertEqual(table.df.at['x', 'ColA'][0].text, "1")

    def test_from_formatted_DataFrame_raises_on_raw_cells(self):
        df = pd.DataFrame({'ColA': {'a': 3.0}})
        with self.assertRaises(ValueError):
            TableData.from_formatted_DataFrame(df)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestFormatAndColorCells(unittest.TestCase):
    """Test format_and_color_cells."""

    def test_default_fmt_is_str(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 3.0}}))
        table.format_and_color_cells()
        self.assertEqual(table.df.at['a', 'ColA'][0].text, str(3.0))

    def test_custom_fmt_applied(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 3.0, 'b': float('nan')}}))
        table.format_and_color_cells(fmt=TableData.fmt_fixed(1, suffix='%'))
        self.assertEqual(table.df.at['a', 'ColA'][0].text, "3.0%")
        self.assertEqual(table.df.at['b', 'ColA'][0].text, "---")

    def test_custom_color_fn_applied(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 3.0, 'b': 1.0}}))
        table.format_and_color_cells(color_fn=lambda v: 'green' if v > 2 else 'red')
        self.assertEqual(table.df.at['a', 'ColA'][0].color, 'green')
        self.assertEqual(table.df.at['b', 'ColA'][0].color, 'red')

    def test_default_color_fn_flags_nan_as_failure(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 3.0, 'b': float('nan')}}))
        table.format_and_color_cells()
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        self.assertEqual(table.df.at['b', 'ColA'][0].color, style.TextFailureColor)
        self.assertEqual(table.df.at['a', 'ColA'][0].color, style.TextColor)

    def test_preserves_raw_value(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 3.0}}))
        table.format_and_color_cells()
        self.assertEqual(table.df.at['a', 'ColA'][0].raw_value, 3.0)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestHighlightBestAndWorstResultsByColumn(unittest.TestCase):
    """Test highlight_best_and_worst_results_by_column."""

    def test_best_value_is_bold_second_is_underlined(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 3.0, 'b': 1.0, 'c': 2.0}}))
        table.highlight_best_and_worst_results_by_column()
        self.assertIn(TableData.TextStyle.BOLD, table.df.at['a', 'ColA'][0].styles)
        self.assertNotIn(TableData.TextStyle.UNDERLINE, table.df.at['a', 'ColA'][0].styles)
        self.assertIn(TableData.TextStyle.UNDERLINE, table.df.at['c', 'ColA'][0].styles)
        self.assertNotIn(TableData.TextStyle.BOLD, table.df.at['c', 'ColA'][0].styles)
        self.assertEqual(table.df.at['b', 'ColA'][0].styles, [])

    def test_higher_is_better_false_flips_ranking(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 3.0, 'b': 1.0, 'c': 2.0}}))
        table.highlight_best_and_worst_results_by_column(higher_is_better=False)
        self.assertIn(TableData.TextStyle.BOLD, table.df.at['b', 'ColA'][0].styles)
        self.assertIn(TableData.TextStyle.UNDERLINE, table.df.at['c', 'ColA'][0].styles)
        self.assertEqual(table.df.at['a', 'ColA'][0].styles, [])

    def test_ties_all_bolded(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 5.0, 'b': 5.0, 'c': 1.0}}))
        table.highlight_best_and_worst_results_by_column()
        self.assertIn(TableData.TextStyle.BOLD, table.df.at['a', 'ColA'][0].styles)
        self.assertIn(TableData.TextStyle.BOLD, table.df.at['b', 'ColA'][0].styles)
        self.assertNotIn(TableData.TextStyle.BOLD, table.df.at['c', 'ColA'][0].styles)
        self.assertIn(TableData.TextStyle.UNDERLINE, table.df.at['c', 'ColA'][0].styles)  # sole remaining value -> 2nd place

    def test_nan_never_bold_or_underlined_even_if_tied(self):
        # NaN values are dropped before ranking, so even a "tie" among all-NaN
        # entries never produces bold/underline.
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': float('nan'), 'b': float('nan'), 'c': 1.0}}))
        table.highlight_best_and_worst_results_by_column()
        self.assertEqual(table.df.at['a', 'ColA'][0].styles, [])
        self.assertEqual(table.df.at['b', 'ColA'][0].styles, [])
        self.assertIn(TableData.TextStyle.BOLD, table.df.at['c', 'ColA'][0].styles)

    def test_single_value_is_bold_only(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 1.0}}))
        table.highlight_best_and_worst_results_by_column()
        self.assertIn(TableData.TextStyle.BOLD, table.df.at['a', 'ColA'][0].styles)
        self.assertNotIn(TableData.TextStyle.UNDERLINE, table.df.at['a', 'ColA'][0].styles)

    def test_custom_rank_styles_bold_only(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 3.0, 'b': 1.0, 'c': 2.0}}))
        table.highlight_best_and_worst_results_by_column(rank_styles=[TableData.TextStyle.BOLD])
        self.assertIn(TableData.TextStyle.BOLD, table.df.at['a', 'ColA'][0].styles)
        self.assertEqual(table.df.at['c', 'ColA'][0].styles, [])  # 2nd place left unstyled without an entry

    def test_custom_rank_styles_multiple_styles_per_rank(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 3.0, 'b': 1.0}}))
        table.highlight_best_and_worst_results_by_column(
            rank_styles=[[TableData.TextStyle.BOLD, TableData.TextStyle.UNDERLINE]])
        self.assertIn(TableData.TextStyle.BOLD, table.df.at['a', 'ColA'][0].styles)
        self.assertIn(TableData.TextStyle.UNDERLINE, table.df.at['a', 'ColA'][0].styles)

    def test_preserves_existing_text_and_color(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 3.0, 'b': 1.0}}))
        table.format_and_color_cells(fmt=TableData.fmt_fixed(1, suffix='%'))
        table.highlight_best_and_worst_results_by_column()
        self.assertEqual(table.df.at['a', 'ColA'][0].text, "3.0%")
        self.assertIn(TableData.TextStyle.BOLD, table.df.at['a', 'ColA'][0].styles)

    def test_format_and_color_cells_after_highlight_resets_styles(self):
        # format_and_color_cells rebuilds each segment from scratch, so it must
        # run before highlight_best_and_worst_results_by_column, not after.
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'a': 3.0, 'b': 1.0}}))
        table.highlight_best_and_worst_results_by_column()
        table.format_and_color_cells()
        self.assertEqual(table.df.at['a', 'ColA'][0].styles, [])


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestMergeTableData(unittest.TestCase):
    """Test merge_TableData."""

    def _make_single_cell_table(self, text, title="T", index_label='x', column='ColA'):
        df = pd.DataFrame({column: [[TableData.FormattedTextSegment(text)]]}, index=[index_label])
        df.attrs["title"] = title
        return TableData.from_formatted_DataFrame(df)

    def _make_two_row_table(self, text_map, title="T", column='ColA'):
        df = pd.DataFrame({column: {k: [TableData.FormattedTextSegment(v)] for k, v in text_map.items()}})
        df.attrs["title"] = title
        return TableData.from_formatted_DataFrame(df)

    def test_merges_two_tables_with_default_separator(self):
        a = self._make_single_cell_table("1")
        b = self._make_single_cell_table("2")
        merged = TableData.merge_TableData(a, b)
        texts = [seg.text for seg in merged.df.at['x', 'ColA']]
        self.assertEqual(texts, ["1", "/", "2"])

    def test_custom_separator(self):
        a = self._make_single_cell_table("1")
        b = self._make_single_cell_table("2")
        merged = TableData.merge_TableData(a, b, separator='-')
        texts = [seg.text for seg in merged.df.at['x', 'ColA']]
        self.assertEqual(texts, ["1", "-", "2"])

    def test_merges_three_tables(self):
        a = self._make_single_cell_table("1")
        b = self._make_single_cell_table("2")
        c = self._make_single_cell_table("3")
        merged = TableData.merge_TableData(a, b, c)
        texts = [seg.text for seg in merged.df.at['x', 'ColA']]
        self.assertEqual(texts, ["1", "/", "2", "/", "3"])

    def test_single_table_is_passthrough_with_no_separator(self):
        a = self._make_single_cell_table("1")
        merged = TableData.merge_TableData(a)
        texts = [seg.text for seg in merged.df.at['x', 'ColA']]
        self.assertEqual(texts, ["1"])

    def test_preserves_all_keys(self):
        a = self._make_two_row_table({'x': "1", 'y': "3"})
        b = self._make_two_row_table({'x': "2", 'y': "4"})
        merged = TableData.merge_TableData(a, b)
        self.assertEqual(set(merged.df.index), {'x', 'y'})

    def test_separator_color_defaults_to_georgia_tech_text_color(self):
        a = self._make_single_cell_table("1")
        b = self._make_single_cell_table("2")
        merged = TableData.merge_TableData(a, b)
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        self.assertEqual(merged.df.at['x', 'ColA'][1].color, style.TextColor)

    def test_separator_color_uses_given_style(self):
        a = self._make_single_cell_table("1")
        b = self._make_single_cell_table("2")
        merged = TableData.merge_TableData(a, b, style=TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        style = TableData.get_table_style(TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(merged.df.at['x', 'ColA'][1].color, style.TextColor)

    def test_merges_output_of_format_and_color_cells(self):
        table_a = TableData.from_DataFrame(pd.DataFrame({'ColA': {'r1': 5.0, 'r2': 3.0}}))
        table_a.format_and_color_cells()
        table_b = TableData.from_DataFrame(pd.DataFrame({'ColA': {'r1': 50.0, 'r2': 30.0}}))
        table_b.format_and_color_cells()
        merged = TableData.merge_TableData(table_a, table_b)
        texts = [seg.text for seg in merged.df.at['r1', 'ColA']]
        self.assertEqual(texts, ["5.0", "/", "50.0"])

    def test_raises_on_index_mismatch(self):
        a = self._make_single_cell_table("1", index_label='x')
        b = self._make_single_cell_table("2", index_label='y')
        with self.assertRaises(ValueError):
            TableData.merge_TableData(a, b)

    def test_raises_on_title_mismatch(self):
        a = self._make_single_cell_table("1", title="A")
        b = self._make_single_cell_table("2", title="B")
        with self.assertRaises(ValueError):
            TableData.merge_TableData(a, b)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestColorFunctions(unittest.TestCase):
    """Test color_fn_NAVY_RED_missing_or_equal / color_fn_NAVY_RED_missing_or_above."""

    def test_missing_or_equal_flags_nan(self):
        color_fn = TableData.color_fn_NAVY_RED_missing_or_equal()
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        self.assertEqual(color_fn(float('nan')), style.TextFailureColor)

    def test_missing_or_equal_flags_equal_value(self):
        color_fn = TableData.color_fn_NAVY_RED_missing_or_equal(equal_value=5.0)
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        self.assertEqual(color_fn(5.0), style.TextFailureColor)
        self.assertEqual(color_fn(0.0), style.TextColor)

    def test_missing_or_equal_default_equal_value_is_zero(self):
        color_fn = TableData.color_fn_NAVY_RED_missing_or_equal()
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        self.assertEqual(color_fn(0.0), style.TextFailureColor)
        self.assertEqual(color_fn(1.0), style.TextColor)

    def test_missing_or_above_flags_nan_and_above_threshold(self):
        color_fn = TableData.color_fn_NAVY_RED_missing_or_above(10.0)
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        self.assertEqual(color_fn(float('nan')), style.TextFailureColor)
        self.assertEqual(color_fn(10.1), style.TextFailureColor)
        self.assertEqual(color_fn(10.0), style.TextColor)
        self.assertEqual(color_fn(-5.0), style.TextColor)

    def test_missing_or_above_infinite_threshold_never_flags_real_values(self):
        color_fn = TableData.color_fn_NAVY_RED_missing_or_above(float('inf'))
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        self.assertEqual(color_fn(1e300), style.TextColor)
        self.assertEqual(color_fn(float('nan')), style.TextFailureColor)

    def test_color_fns_respect_style_parameter(self):
        style = TableData.get_table_style(TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        eq_fn = TableData.color_fn_NAVY_RED_missing_or_equal(style=TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        above_fn = TableData.color_fn_NAVY_RED_missing_or_above(10.0, style=TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(eq_fn(1.0), style.TextColor)
        self.assertEqual(above_fn(1.0), style.TextColor)

    def test_color_fns_return_independent_closures(self):
        # Two calls with different thresholds must not share state.
        fn_low = TableData.color_fn_NAVY_RED_missing_or_above(1.0)
        fn_high = TableData.color_fn_NAVY_RED_missing_or_above(100.0)
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        self.assertEqual(fn_low(5.0), style.TextFailureColor)
        self.assertEqual(fn_high(5.0), style.TextColor)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestFmtFixed(unittest.TestCase):
    """Test fmt_fixed."""

    def test_default_precision(self):
        fmt = TableData.fmt_fixed()
        self.assertEqual(fmt(3.14159), "3.14")

    def test_custom_precision(self):
        fmt = TableData.fmt_fixed(precision=0)
        self.assertEqual(fmt(3.9), "4")

    def test_suffix_applied(self):
        fmt = TableData.fmt_fixed(precision=1, suffix='%')
        self.assertEqual(fmt(50.0), "50.0%")

    def test_nan_uses_missing_str(self):
        fmt = TableData.fmt_fixed(missing_str="N/A")
        self.assertEqual(fmt(float('nan')), "N/A")

    def test_default_missing_str_is_dashes(self):
        fmt = TableData.fmt_fixed()
        self.assertEqual(fmt(float('nan')), "---")


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestRenderOntoAx(unittest.TestCase):
    """Test TableData.render_onto_ax."""

    def _make_table(self, values=None, title="Method"):
        values = values if values is not None else {'ColA': {'r1': 5.0, 'r2': 3.0, 'r3': 1.0}}
        df = pd.DataFrame(values)
        df.attrs["title"] = title
        table = TableData.from_DataFrame(df)
        table.format_and_color_cells()
        table.highlight_best_and_worst_results_by_column()
        return table

    def setUp(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 3))
        self.ax.axis('off')
        self.fig.tight_layout(pad=0.0)

    def tearDown(self):
        plt.close(self.fig)

    def test_renders_without_raising(self):
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax)

    def test_missing_title_raises_key_error(self):
        table = TableData.from_DataFrame(pd.DataFrame({'ColA': {'r1': 1.0}}))
        with self.assertRaises(KeyError):
            table.render_onto_ax(self.fig, self.ax)

    def test_column_labels_include_title_and_columns(self):
        table = self._make_table(title="MyTitle")
        table.render_onto_ax(self.fig, self.ax)
        tbl = [c for c in self.ax.tables][0]
        self.assertEqual(tbl[0, 0].get_text().get_text(), "MyTitle")
        self.assertEqual(tbl[0, 1].get_text().get_text(), "ColA")

    def test_row_labels_are_index_values(self):
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax)
        tbl = [c for c in self.ax.tables][0]
        self.assertEqual(tbl[1, 0].get_text().get_text(), "r1")
        self.assertEqual(tbl[2, 0].get_text().get_text(), "r2")
        self.assertEqual(tbl[3, 0].get_text().get_text(), "r3")

    def test_header_facecolor_matches_style(self):
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax)
        tbl = [c for c in self.ax.tables][0]
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        self.assertEqual(tbl[0, 0].get_facecolor(), to_rgba(style.HeaderColor))

    def test_header_facecolor_matches_given_style(self):
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax, style=TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        tbl = [c for c in self.ax.tables][0]
        style = TableData.get_table_style(TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(tbl[0, 0].get_facecolor(), to_rgba(style.HeaderColor))

    def test_row_facecolors_alternate(self):
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax)
        tbl = [c for c in self.ax.tables][0]
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        self.assertEqual(tbl[1, 0].get_facecolor(), to_rgba(style.RowColors[0]))
        self.assertEqual(tbl[2, 0].get_facecolor(), to_rgba(style.RowColors[1]))
        self.assertEqual(tbl[3, 0].get_facecolor(), to_rgba(style.RowColors[0]))

    def test_header_text_color_matches_style(self):
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax, style=TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        tbl = [c for c in self.ax.tables][0]
        style = TableData.get_table_style(TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(tbl[0, 1].get_text().get_color(), style.HeaderTextColor)

    def test_row_label_text_color_matches_style_text_color(self):
        """Regression test: row labels (method names) must use the style's
        TextColor, not matplotlib's default text color."""
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax, style=TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        tbl = [c for c in self.ax.tables][0]
        style = TableData.get_table_style(TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
        self.assertEqual(tbl[1, 0].get_text().get_color(), style.TextColor)
        self.assertEqual(tbl[2, 0].get_text().get_color(), style.TextColor)
        self.assertEqual(tbl[3, 0].get_text().get_color(), style.TextColor)

    def test_data_font_size_overrides_font_size(self):
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax, font_size=11, data_font_size=20)
        tbl = [c for c in self.ax.tables][0]
        self.assertEqual(tbl[0, 1].get_text().get_fontsize(), 11)  # header keeps font_size
        self.assertEqual(tbl[1, 1].get_text().get_fontsize(), 20)  # data cell overridden

    def test_simple_cell_text_and_bold_applied(self):
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax)
        tbl = [c for c in self.ax.tables][0]
        # r1 (5.0) is the best value -> bold, non-underlined -> rendered directly in-cell.
        self.assertEqual(tbl[1, 1].get_text().get_text(), "5.0")
        self.assertEqual(tbl[1, 1].get_text().get_fontweight(), 'bold')

    def test_underlined_cell_text_cleared_and_redrawn_via_fig_text(self):
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax)
        tbl = [c for c in self.ax.tables][0]
        # r2 (3.0) is second-best -> underlined -> in-cell text wiped, redrawn as fig.text.
        self.assertEqual(tbl[2, 1].get_text().get_text(), "")
        redrawn_texts = [t.get_text() for t in self.fig.texts]
        self.assertIn("3.0", redrawn_texts)

    def test_multi_segment_cell_produces_one_fig_text_per_segment(self):
        table_a = self._make_table(values={'ColA': {'r1': 5.0}})
        table_b = self._make_table(values={'ColA': {'r1': 50.0}})
        merged = TableData.merge_TableData(table_a, table_b)
        merged.render_onto_ax(self.fig, self.ax)
        redrawn_texts = sorted(t.get_text() for t in self.fig.texts)
        self.assertEqual(redrawn_texts, ["/", "5.0", "50.0"])

    def test_heavy_divider_before_called_with_zero_based_column_indices(self):
        table = self._make_table(values={'ColA': {'r1': 1.0}, 'ColB': {'r1': 2.0}, 'ColC': {'r1': 3.0}})
        heavy_fn = unittest.mock.Mock(return_value=False)
        table.render_onto_ax(self.fig, self.ax, heavy_divider_before=heavy_fn)
        self.assertEqual(heavy_fn.call_args_list, [unittest.mock.call(0), unittest.mock.call(1), unittest.mock.call(2)])

    def test_heavy_divider_is_white_but_thicker(self):
        table = self._make_table(values={'ColA': {'r1': 1.0}, 'ColB': {'r1': 2.0}})
        table.render_onto_ax(self.fig, self.ax, heavy_divider_before=lambda _: True)
        divider_lines = [line for line in self.fig.artists if isinstance(line, mlines.Line2D)]
        self.assertTrue(divider_lines)
        for line in divider_lines:
            self.assertEqual(line.get_color(), 'white')
            self.assertEqual(line.get_linewidth(), 2.2)

    def test_non_heavy_divider_is_white_and_thin(self):
        table = self._make_table(values={'ColA': {'r1': 1.0}, 'ColB': {'r1': 2.0}})
        table.render_onto_ax(self.fig, self.ax, heavy_divider_before=lambda _: False)
        divider_lines = [line for line in self.fig.artists if isinstance(line, mlines.Line2D)]
        self.assertTrue(divider_lines)
        for line in divider_lines:
            self.assertEqual(line.get_color(), 'white')
            self.assertEqual(line.get_linewidth(), 1.0)

    def test_default_tbl_bbox_is_full_axes(self):
        table = self._make_table()
        # Should not raise, and should use the full [0, 0, 1, 1] axes bbox by default.
        table.render_onto_ax(self.fig, self.ax, tbl_bbox=None)
        tbl = [c for c in self.ax.tables][0]
        self.assertIsNotNone(tbl)

    def test_custom_tbl_bbox_accepted(self):
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax, tbl_bbox=[0, 0.3, 0.75, 0.3])

    def test_overlay_artists_use_figure_fraction_transform_not_identity(self):
        """Regression test: overlay text/lines must use fig.transFigure, not raw
        display pixels, so they stay correctly positioned after a later
        ``savefig(bbox_inches='tight')`` recrops the canvas (see bug where
        underlines/dividers rendered in the wrong place entirely)."""
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax)

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
        table = self._make_table()
        table.render_onto_ax(self.fig, self.ax)

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
class TestToPdf(unittest.TestCase):
    """Test TableData.to_pdf."""

    def _make_table(self, title="Method"):
        df = pd.DataFrame({'ColA': {'r1': 5.0, 'r2': 3.0, 'r3': 1.0}})
        df.attrs["title"] = title
        table = TableData.from_DataFrame(df)
        table.format_and_color_cells()
        table.highlight_best_and_worst_results_by_column()
        return table

    def test_saves_single_table_pdf(self):
        table = self._make_table()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "out.pdf"
            TableData.to_pdf([table], str(save_path))
            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 0)

    def test_saves_multiple_tables_pdf(self):
        table1 = self._make_table(title="Table1")
        table2 = self._make_table(title="Table2")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "out.pdf"
            TableData.to_pdf([table1, table2], str(save_path))
            self.assertTrue(save_path.exists())

    def test_saves_as_png(self):
        table = self._make_table()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "out.png"
            TableData.to_pdf([table], str(save_path))
            self.assertTrue(save_path.exists())

    def test_creates_missing_parent_directory(self):
        table = self._make_table()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dir" / "out.pdf"
            # to_pdf itself doesn't mkdir; verify caller-created parent dirs
            # are respected and no unrelated error is raised when the
            # directory already exists.
            save_path.parent.mkdir(parents=True, exist_ok=True)
            TableData.to_pdf([table], str(save_path))
            self.assertTrue(save_path.exists())

    def test_style_forwarded_to_render_onto_ax(self):
        table = self._make_table()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "out.pdf"
            with unittest.mock.patch.object(table, 'render_onto_ax', wraps=table.render_onto_ax) as spy:
                TableData.to_pdf([table], str(save_path), style=TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)
                _, kwargs = spy.call_args
                self.assertEqual(kwargs['style'], TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY)

    def test_default_style_is_georgia_tech(self):
        table = self._make_table()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "out.pdf"
            with unittest.mock.patch.object(table, 'render_onto_ax', wraps=table.render_onto_ax) as spy:
                TableData.to_pdf([table], str(save_path))
                _, kwargs = spy.call_args
                self.assertEqual(kwargs['style'], TableData.TableStyleName.GEORGIA_TECH)


if __name__ == '__main__':
    unittest.main()
