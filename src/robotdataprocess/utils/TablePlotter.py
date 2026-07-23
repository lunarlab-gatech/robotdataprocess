from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.font_manager import FontProperties
from matplotlib.table import Table
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Any

class TablePlotter:
    """Resolve DataFrame values into styled FormattedTextSegments and draw them as matplotlib tables."""

    @dataclass(frozen=True)
    class FormattedTextSegment():
        """One styled segment of text within a table cell."""
        text: str
        bold: bool = False
        underline: bool = False
        color: Optional[str] = None

    # =========================================================================
    # ======================== Table Style Functions ==========================
    # =========================================================================
    class TableStyleName(Enum):
        """Supported named table styles for ``get_table_style``."""
        GEORGIA_TECH = "GeorgiaTech"
        BRIGHAM_YOUNG_UNIVERSITY = "BrighamYoungUniversity"

    @dataclass(frozen=True)
    class TableStyle():
        """Colors used to draw a styled table."""
        HeaderColor: str
        HeaderTextColor: str
        TextColor: str
        TextFailureColor: str
        RowColors: Tuple[str, str]

    @staticmethod
    def get_table_style(style: TablePlotter.TableStyleName) -> TablePlotter.TableStyle:
        """Return the TableStyle for a named style. Only ``GEORGIA_TECH`` is currently supported."""
        if style is TablePlotter.TableStyleName.GEORGIA_TECH:
            return TablePlotter.TableStyle(
                HeaderColor='#B5A060', HeaderTextColor='#FFFFFF', TextColor='#1A3055', TextFailureColor='#CC2222',
                RowColors=('#EDEADE', '#D8D3C3'),
            )
        elif style is TablePlotter.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY:
            return TablePlotter.TableStyle(
                HeaderColor="#2F61B3", HeaderTextColor='#FFFFFF', TextColor="#002E5D", TextFailureColor='#9E2A2B',
                RowColors=("#DBDEE8", "#C9CEDE"),
            )
        else:
            raise ValueError(f"Unsupported table style: {style}")

    # =========================================================================
    # ========================= pd.Series functions ===========================
    # =========================================================================
    @staticmethod
    def _rank_data_in_Series(data: pd.Series, higher_is_better: bool = True) -> List[set[int]]:
        """Return label groups ordered best-to-worst, with ties in the same group.

        NaN values are excluded entirely — they never appear in any group, so
        they're never bolded/underlined.
        """
        data = data.dropna()
        groups: List = []
        for v in sorted(data.unique(), reverse=higher_is_better):
            groups.append(set(data[data == v].index))
        return groups

    @staticmethod
    def convert_numbers_to_TextSegments(data: pd.Series, color_fn: Optional[Callable[[float], str]] = None,
        fmt: Optional[Callable[[float], str]] = str, emphasize_rankings: bool = True, higher_is_better: bool = True) -> pd.Series:
        """
        Resolve a single row or column of raw numeric values into styled, formatted TextSegments.

        Ranking and display formatting are both derived from the same raw
        numeric values, so callers don't need to build a separately formatted
        display Series alongside a raw rank Series.

        For cells that display more than one ranked value (e.g. "42/50" ranked
        independently per side), call this once per value and combine the
        results with ``merge_Series``.

        Args:
            data: Raw numeric values to display and rank, indexed by label.
            color_fn: callable(value: float) -> str; maps a value to a text
                color (e.g. red above a threshold, green below zero). Defaults
                to ``color_fn_NAVY_RED_missing_or_equal()`` (red for NaN or 0.0).
            fmt: callable(value: float) -> str; formats a value for display.
                Defaults to ``str``.
            emphasize_rankings: If False, skip ranking entirely — no bold/underline,
                only color.
            higher_is_better: If False, the lowest value(s) in the column are
                bolded/underlined instead of the highest (e.g. for ATE, RPE,
                runtime, or data size, where lower is better).

        Returns:
            Series keyed by ``data``'s index label, holding a single-element
            list containing one TextSegment.
        """
        # Fill in default arguments that can't be default parameters
        if color_fn is None:
            color_fn = TablePlotter.color_fn_NAVY_RED_missing_or_equal()

        # Determine which values to bold and underline
        bold_labels = set()
        ul_labels = set()
        if emphasize_rankings:
            groups: List[set[int]] = TablePlotter._rank_data_in_Series(data, higher_is_better)
            bold_labels = groups[0] if len(groups) >= 1 else set()
            ul_labels = (groups[1] if len(groups) >= 2 else set()) - bold_labels

        # For each cell, create a FormattedTextSegment
        segments: Dict[Any, List[TablePlotter.FormattedTextSegment]] = {}
        for index, value in data.items():
            bold: bool = index in bold_labels
            underline: bool = index in ul_labels
            segments[index] = [TablePlotter.FormattedTextSegment(
                fmt(value), bold=bold, underline=underline, color=color_fn(value)
            )]
        return pd.Series(segments)

    @staticmethod
    def merge_Series(*segment_maps: pd.Series, separator: str = '/',
        style: TablePlotter.TableStyleName = TableStyleName.GEORGIA_TECH) -> pd.Series:
        """
        Combine multiple resolve_table_segments() outputs into one cell each, joined by a separator.

        Args:
            *segment_maps: Two or more outputs of ``resolve_table_segments``, one
                per value shown in the cell (e.g. one for "42", one for "50").
            separator: Text placed between each value's segments (e.g. "/").
            style: Named table style whose ``TextColor`` is used for the separator.

        Returns:
            Series keyed by label, holding the combined TextSegment list.
        """
        separator_segment = TablePlotter.FormattedTextSegment(separator, color=TablePlotter.get_table_style(style).TextColor)
        merged: Dict[str, List[TablePlotter.FormattedTextSegment]] = {}
        for key in segment_maps[0].keys():
            combined: List[TablePlotter.FormattedTextSegment] = []
            for i, segment_map in enumerate(segment_maps):
                if i > 0:
                    combined.append(separator_segment)
                combined.extend(segment_map[key])
            merged[key] = combined
        return pd.Series(merged)

    # =========================================================================
    # =========================== Color Functions =============================
    # =========================================================================

    @staticmethod
    def color_fn_NAVY_RED_missing_or_equal(equal_value: float = 0.0,
        style: TablePlotter.TableStyleName = TableStyleName.GEORGIA_TECH) -> Callable[[float], str]:
        """Build a color_fn: red for NaN (missing) or values equal to equal_value, navy otherwise."""
        resolved_style = TablePlotter.get_table_style(style)

        def color_fn(value: float) -> str:
            return resolved_style.TextFailureColor if math.isnan(value) or value == equal_value else resolved_style.TextColor
        return color_fn

    @staticmethod
    def color_fn_NAVY_RED_missing_or_above(threshold: float,
        style: TablePlotter.TableStyleName = TableStyleName.GEORGIA_TECH) -> Callable[[float], str]:
        """Build a color_fn: red for NaN (missing) or values above threshold, navy otherwise."""
        resolved_style = TablePlotter.get_table_style(style)

        def color_fn(value: float) -> str:
            return resolved_style.TextFailureColor if math.isnan(value) or value > threshold else resolved_style.TextColor
        return color_fn

    # =========================================================================
    # =========================== Format Functions ============================
    # =========================================================================

    @staticmethod
    def fmt_fixed(precision: int = 2, suffix: str = "", missing_str: str = "---") -> Callable[[float], str]:
        """
        Build a fmt: NaN renders as missing_str, otherwise fixed-point with precision decimals plus suffix.
        Use ``precision=0`` for plain integer display (e.g. counts).
        """
        def fmt(value: float) -> str:
            return missing_str if math.isnan(value) else f"{value:.{precision}f}{suffix}"
        return fmt

    # =========================================================================
    # ========================= Plotting functions ============================
    # =========================================================================
    @staticmethod
    def render_table_onto_ax(fig: plt.Figure, ax: Axes, df: pd.DataFrame, tbl_bbox: Optional[List[float]] = None,
        font_size: int = 11, data_font_size: Optional[int] = None,
        heavy_divider_before: Callable[[int], bool] = lambda _: False,
        style: TablePlotter.TableStyleName = TableStyleName.GEORGIA_TECH) -> None:
        """
        Render one styled table onto ax, including its post-render decorations.

        The figure layout must be finalized (e.g. via ``tight_layout`` or fixed
        ``GridSpec`` margins) before calling this, since positioning multi-segment
        cell text, column dividers, and underlines requires renderer-level
        bounding boxes that only exist once the layout is final. Call this once
        per table; each call finalizes its own table independently, forcing its
        own ``fig.canvas.draw()`` passes to obtain those bounding boxes.

        The caller is responsible for calling ``ax.axis('off')`` itself, and
        for doing so *before* finalizing the layout (e.g. before
        ``tight_layout()``) — hiding axis chrome after layout has already been
        computed doesn't shrink the margins that were reserved for it.

        Args:
            fig: The matplotlib figure that owns ax.
            ax: The axes to render the table onto.
            df: DataFrame whose cells each hold a ``List[TablePlotter.FormattedTextSegment]``
                (as produced by ``convert_numbers_to_TextSegments``/``merge_Series``),
                with the table's title in ``df.attrs["title"]``.
            tbl_bbox: ``[x0, y0, width, height]`` bbox (axes coordinates) passed
                to ``ax.table``; controls how much of the axes the table fills.
                Defaults to ``[0, 0, 1, 1]`` (full axes).
            font_size: Header/data font size.
            data_font_size: Overrides just the data font size.
            heavy_divider_before: callable(col_idx: int) -> bool; if True for a
                given 0-based index into df.columns, the divider before that
                column is drawn thicker than the plain white dividers
                separating ordinary data columns (e.g. to set off a summary
                column). Defaults to no heavy dividers.
            style: Named table style used for header, row, and divider colors.
        """

        # Fill in default arguments that can't be default parameters
        if tbl_bbox is None:
            tbl_bbox = [0, 0, 1, 1]
        resolved_style = TablePlotter.get_table_style(style)

        # Set column titles
        col_labels: List[str] = [df.attrs["title"]] + list(df.columns)

        # Get rows and data
        cell_text: List[List[str]] = [
            [str(row_name)] + [''.join(seg.text for seg in segments) for segments in row]
            for row_name, row in df.iterrows()
        ]

        # Extract number of rows and columns
        n_rows: int = len(cell_text)
        n_cols: int = len(col_labels)

        # Create a matplotlib Table with text aligned in center
        tbl: Table = ax.table(cellText=cell_text, colLabels=col_labels,
                              bbox=tbl_bbox, cellLoc='center')
        tbl.auto_set_column_width(col=list(range(n_cols)))

        # Set the font sizes
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(font_size)
        if data_font_size is not None:
            for i in range(1, n_rows + 1):
                for j in range(1, n_cols):
                    tbl[i, j].get_text().set_fontsize(data_font_size)

        # Halve the first column width
        for row_i in range(n_rows + 1):
            cell = tbl[row_i, 0]
            cell.set_width(cell.get_width() * 0.5)

        # Set the Header styling
        for j in range(n_cols):
            cell = tbl[0, j]
            cell.set_facecolor(resolved_style.HeaderColor)
            cell.set_edgecolor('none')
            cell.get_text().set_color(resolved_style.HeaderTextColor)
            cell.get_text().set_fontweight('bold')

        # Set the row styling
        for i in range(n_rows):
            face_color: str = resolved_style.RowColors[i % 2]
            for j in range(n_cols):
                cell = tbl[i + 1, j]
                cell.set_facecolor(face_color)
                cell.set_edgecolor('none')

                # Make first column left justified
                if j == 0:
                    cell.get_text().set_ha('left')
                    cell.get_text().set_color(resolved_style.TextColor)

        # Draw simple TextSegments into Cells, save more complicated once for after drawing
        post_render_cells: List[Tuple[int, int, List[TablePlotter.FormattedTextSegment]]] = []
        for i, (_, row) in enumerate(df.iterrows()):
            it = i + 1
            for j, segments in enumerate(row):
                segments: List[TablePlotter.FormattedTextSegment]
                jt = j + 1

                # In the case of a single non-underlined segment, update it now
                if len(segments) == 1 and not segments[0].underline:
                    seg = segments[0]
                    cell = tbl[it, jt]
                    if seg.color is not None:
                        cell.get_text().set_color(seg.color)
                    if seg.bold:
                        cell.get_text().set_fontweight('bold')

                # Otherwise, wipe previous text and save to update after post-render
                # (So that final locations of text are locked in)
                else:
                    tbl[it, jt].get_text().set_text('')
                    post_render_cells.append((it, jt, segments))

        # Post-render decorations: multi-segment cell text, column dividers, and
        # underlines all need a renderer to measure/position against.
        fig.canvas.draw()
        renderer: RendererBase = fig.canvas.get_renderer()

        # Draw more complicated TextSegments into cells
        underline_texts: List[Text] = []
        for it, jt, segments in post_render_cells:

            # Get the bounding box of the relevant cell
            cell = tbl[it, jt]
            bb: Bbox = cell.get_window_extent(renderer=renderer)

            # Get widths of the text segmetns
            widths: List[float] = []
            for seg in segments:
                # Determine font properties for this text segment
                fontweight = 'bold' if seg.bold else 'normal'
                fontstyle = 'italic' if seg.underline else 'normal'
                fp = FontProperties(size=font_size, weight=fontweight, style=fontstyle)

                # Get and store the width of this text segment
                w, _, _ = renderer.get_text_width_height_descent(seg.text, fp, ismath=False)
                widths.append(w)

            # Get initial x_cursor position for writing text
            cx: float = (bb.x0 + bb.x1) / 2
            cy: float = (bb.y0 + bb.y1) / 2
            x_cursor: float = cx - sum(widths) / 2

            # Draw text piece by piece. Positions are converted from display
            # pixels (this draw pass) to figure-fraction coordinates so they
            # stay correct even if a later ``savefig(bbox_inches='tight')``
            # rescales the saved canvas — display pixels from this pass would
            # otherwise go stale.
            for k, seg in enumerate(segments):
                fig_x, fig_y = fig.transFigure.inverted().transform((x_cursor + widths[k] / 2, cy))
                txt: Text = fig.text(fig_x, fig_y, seg.text,
                                     transform=fig.transFigure,
                                     fontsize=font_size, color=seg.color,
                                     fontweight='bold' if seg.bold else 'normal',
                                     fontstyle='italic' if seg.underline else 'normal',
                                     ha='center', va='center')
                if seg.underline: # Remember to underline this after it is drawn later
                    underline_texts.append(txt)
                x_cursor += widths[k]

        # Drawn dividers between each column
        for j in range(1, n_cols):
            header_bb: Bbox = tbl[0, j].get_window_extent(renderer=renderer)
            bottom_bb: Bbox = tbl[n_rows, j].get_window_extent(renderer=renderer)
            is_heavy = heavy_divider_before(j - 1)
            fig_x, fig_y0 = fig.transFigure.inverted().transform((bottom_bb.x0, bottom_bb.y0))
            fig_y1 = fig.transFigure.inverted().transform((0, header_bb.y1))[1]
            fig.add_artist(mlines.Line2D(
                [fig_x, fig_x], [fig_y0, fig_y1],
                transform=fig.transFigure, color='white',
                linewidth=2.2 if is_heavy else 1.0, clip_on=False
            ))

        # Drawn any underlines
        if underline_texts:
            fig.canvas.draw()
            for txt in underline_texts:
                bb: Bbox = txt.get_window_extent(renderer=renderer)
                fig_bb: Bbox = bb.transformed(fig.transFigure.inverted())
                fig.add_artist(mlines.Line2D(
                    [fig_bb.x0, fig_bb.x1], [fig_bb.y0, fig_bb.y0],
                    transform=fig.transFigure, color=resolved_style.TextColor,
                    linewidth=0.8, clip_on=False
                ))

    @staticmethod
    def plot_tables_on_pdf(tables: List[pd.DataFrame], save_path: str, row_height: float = 2.4, h_pad: float = 1.2,
        heavy_divider_before: Callable[[int], bool] = lambda _: False,
        style: TablePlotter.TableStyleName = TableStyleName.GEORGIA_TECH) -> None:
        """
        Render a list of styled tables on a single PDF.

        Args:
            tables: List of DataFrames, each whose cells hold a
                ``List[TablePlotter.FormattedTextSegment]`` (as produced by
                ``convert_numbers_to_TextSegments``/``merge_Series``), with
                the table's title in ``df.attrs["title"]``.
            save_path: Output file path (PDF or PNG).
            row_height: Figure height per table in inches.
            h_pad: Vertical padding between subplots passed to tight_layout. The outer
                figure pad is fixed at 0.1 font-size units to minimise top/bottom margins.
            heavy_divider_before: callable(col_idx: int) -> bool; forwarded to
                ``render_table_onto_ax`` for every table (e.g. to set off a
                trailing summary column). Defaults to no heavy dividers.
            style: Named table style used for header, row, and divider colors.
        """

        # Generate a figure with subplots equal to the number of tables
        fig, axes = plt.subplots(len(tables), 1, figsize=(12, row_height * len(tables)))
        axes: List[Axes] = [axes] if len(tables) == 1 else list(axes)

        # Disable matplotlib axis edges
        for ax in axes:
            ax.axis('off')
        fig.tight_layout(pad=0.0, h_pad=h_pad)

        # Render the tables onto the figure we just made
        for ax, df in zip(axes, tables):
            TablePlotter.render_table_onto_ax(fig, ax, df, heavy_divider_before=heavy_divider_before, style=style)

        # Save the figure
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"\nTables saved to {save_path}")
