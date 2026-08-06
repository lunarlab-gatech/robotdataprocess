from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from typeguard import typechecked
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

@typechecked
class TableData:
    """Tracks tabular data and draws/exports it as a matplotlib table or LaTeX table."""

    df: pd.DataFrame # Every cell holds a ``List[TableData.FormattedTextSegment]

    class TextStyle(Enum):
        """Emphasis that can be applied to a ranked cell by ``highlight_best_and_worst_results_by_column``."""
        BOLD = "Bold"
        UNDERLINE = "Underline"

    @dataclass(frozen=True)
    class FormattedTextSegment():
        """One styled segment of text within a table cell."""
        text: str
        styles: List[TableData.TextStyle] = field(default_factory=list)
        color: Optional[str] = None
        raw_value: Optional[Union[float, int, str]] = None

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def _normalize_cell(value: Union[float, int, str]) -> List[TableData.FormattedTextSegment]:
        """Wrap a single raw value as a single plain-text FormattedTextSegment, preserving the raw value."""
        if isinstance(value, list):
            raise ValueError(f"TableData cells must hold a single raw value, not a list; got {value!r}.")
        return [TableData.FormattedTextSegment(str(value), raw_value=value)]

    # =========================================================================
    # ============================ Import Methods ==============================
    # =========================================================================
    @classmethod
    def from_DataFrame(cls, df: pd.DataFrame) -> TableData:
        """
        Build a TableData from a DataFrame of raw values (numbers, strings, etc.), with the
        table's title in ``df.attrs["title"]``. Each cell's raw value is wrapped into a
        ``List[TableData.FormattedTextSegment]``, keeping the raw value on the segment's
        ``raw_value`` so ``highlight_best_and_worst_results`` can later rank/format/color it.
        """
        # object dtype first -- assigning a list into a numeric-dtype cell via .iat[]
        # gets silently unwrapped by pandas (e.g. ["x"] collapses to "x").
        # Positional (.iat) rather than label (.at) access since the index may
        # hold repeated labels (e.g. a multirow grouping column).
        normalized = df.astype(object)
        n_rows, n_cols = normalized.shape
        for row_pos in range(n_rows):
            for col_pos in range(n_cols):
                normalized.iat[row_pos, col_pos] = TableData._normalize_cell(normalized.iat[row_pos, col_pos])
        normalized.attrs = df.attrs
        return cls(normalized)

    @classmethod
    def from_formatted_DataFrame(cls, df: pd.DataFrame) -> TableData:
        """
        Build a TableData from a DataFrame whose cells already hold
        ``List[TableData.FormattedTextSegment]`` (e.g. the output of
        ``merge_TableData``), with the table's title in ``df.attrs["title"]``.
        """
        n_rows, n_cols = df.shape
        for row_pos in range(n_rows):
            for col_pos in range(n_cols):
                value = df.iat[row_pos, col_pos]
                if not (isinstance(value, list) and all(isinstance(v, TableData.FormattedTextSegment) for v in value)):
                    raise ValueError(
                        f"from_formatted_DataFrame expects every cell to hold a List[TableData.FormattedTextSegment], "
                        f"but cell at row {row_pos} ({df.index[row_pos]!r}), column {col_pos} ({df.columns[col_pos]!r}) "
                        f"holds {value!r}. Use from_DataFrame instead."
                    )
        return cls(df)

    # =========================================================================
    # ======================== Table Style Functions ==========================
    # =========================================================================
    class TableStyleName(Enum):
        """Supported named table styles for ``get_table_style``."""
        GEORGIA_TECH = "GeorgiaTech"
        BRIGHAM_YOUNG_UNIVERSITY = "BrighamYoungUniversity"
        LATEX = "Latex"
        TONI_KENSA = "ToniKensa"
        OVIEDO_HIGH_SCHOOL = "OviedoHighSchool"

    @dataclass(frozen=True)
    class TableStyle():
        """Colors used to draw a styled table."""
        HeaderColor: Optional[str]
        HeaderTextColor: Optional[str]
        TextColor: Optional[str]
        TextFailureColor: str
        RowColors: Optional[Tuple[str, str]]

    @staticmethod
    def get_table_style(style: TableData.TableStyleName) -> TableData.TableStyle:
        """Return the TableStyle for a named style."""
        if style is TableData.TableStyleName.GEORGIA_TECH:
            return TableData.TableStyle(
                HeaderColor='#B5A060', HeaderTextColor='#FFFFFF', TextColor='#1A3055', TextFailureColor='#CC2222',
                RowColors=('#EDEADE', '#D8D3C3'),
            )
        elif style is TableData.TableStyleName.BRIGHAM_YOUNG_UNIVERSITY:
            return TableData.TableStyle(
                HeaderColor="#2F61B3", HeaderTextColor='#FFFFFF', TextColor="#002E5D", TextFailureColor='#9E2A2B',
                RowColors=("#DBDEE8", "#C9CEDE"),
            )
        elif style is TableData.TableStyleName.LATEX:
            return TableData.TableStyle(
                HeaderColor=None, HeaderTextColor=None, TextColor=None, TextFailureColor='#CC2222',
                RowColors=None,
            )
        elif style is TableData.TableStyleName.TONI_KENSA:
            return TableData.TableStyle(
                HeaderColor="#404040", HeaderTextColor='#FFFFFF', TextColor='#1A1A1A', TextFailureColor='#D2001F',
                RowColors=("#E7E7E7", "#D8D8D8"),
            )
        elif style is TableData.TableStyleName.OVIEDO_HIGH_SCHOOL:
            return TableData.TableStyle(
                HeaderColor="#C87F36", HeaderTextColor="#FFFFFF", TextColor='#141414', TextFailureColor='#CC2222',
                RowColors=("#FFEFDC", '#FCE0C0'),
            )
        else:
            raise ValueError(f"Unsupported table style: {style}")

    # =========================================================================
    # ======================= Manipulation functions ==========================
    # =========================================================================
    def set_title(self, title: str) -> None:
        """Overwrite this table's title (``df.attrs["title"]``) with ``title``."""
        self.df.attrs["title"] = title

    @staticmethod
    def _rank_data_in_Series(data: pd.Series, higher_is_better: bool = True) -> List[Set[int]]:
        """Return groups of integer positions into ``data`` ordered best-to-worst, with ties in the same group.

        Positions, not labels, since ``data``'s index may hold repeated labels
        (e.g. a multirow grouping column) and so can't identify a row uniquely.
        NaN values are excluded entirely — they never appear in any group, so
        they're never bolded/underlined.
        """
        positions_by_value: Dict[float, Set[int]] = {}
        for pos, value in enumerate(data):
            if pd.isna(value):
                continue
            positions_by_value.setdefault(value, set()).add(pos)
        return [positions_by_value[v] for v in sorted(positions_by_value, reverse=higher_is_better)]

    def format_and_color_cells(self, color_fn: Optional[Callable[[float], Optional[str]]] = None,
        fmt: Optional[Callable[[float], str]] = str) -> None:
        """
        Format and color every cell independently, using each cell's stored raw value.

        Unlike ranking-based emphasis, formatting and coloring only ever look
        at one cell at a time — no comparison across a column is needed to
        pick a display string or a color. ``df.attrs["title"]``, row labels,
        and column names are left untouched.

        Args:
            color_fn: callable(value: float) -> str; maps a value to a text
                color (e.g. red above a threshold, green below zero). Defaults
                to ``color_fn_NAVY_RED_missing_or_equal()`` (red for NaN or 0.0).
            fmt: callable(value: float) -> str; formats a value for display.
                Defaults to ``str``.
        """
        # Fill in default arguments that can't be default parameters
        if color_fn is None:
            color_fn = TableData.color_fn_NAVY_RED_missing_or_equal()

        # Format and color each cell independently (positional: the index may hold
        # repeated labels, e.g. a multirow grouping column)
        n_rows, n_cols = self.df.shape
        for row_pos in range(n_rows):
            for col_pos in range(n_cols):
                value = self.df.iat[row_pos, col_pos][0].raw_value
                self.df.iat[row_pos, col_pos] = [TableData.FormattedTextSegment(
                    fmt(value), color=color_fn(value), raw_value=value
                )]

    def highlight_best_and_worst_results_by_column(self, higher_is_better: bool = True,
        rank_styles: Optional[List[Union[TableData.TextStyle, List[TableData.TextStyle]]]] = None) -> None:
        """
        Style the top-ranked value(s) in each column (e.g. bold 1st place, underline 2nd place), independently per column.

        Ranking is derived from each cell's raw value (kept on
        ``FormattedTextSegment.raw_value``), so callers don't need to build a
        separately formatted display DataFrame alongside a raw rank DataFrame.
        Each cell's existing text and color (see ``format_and_color_cells``)
        are preserved — only ``styles`` is updated. Row labels, column names,
        and ``df.attrs["title"]`` are also left untouched.

        For cells that should display more than one ranked value (e.g.
        "42/50" ranked independently per side), call this once per source
        table and combine the results with ``merge_TableData``.

        Args:
            higher_is_better: If False, the lowest value(s) in each column are
                ranked 1st instead of the highest (e.g. for ATE, RPE, runtime,
                or data size, where lower is better).
            rank_styles: One style (or list of styles) per rank place,
                best-to-worst. Defaults to ``[TextStyle.BOLD, TextStyle.UNDERLINE]``
                (1st place bolded, 2nd place underlined). Pass e.g.
                ``[TextStyle.BOLD]`` to style only 1st place, or
                ``[[TextStyle.BOLD, TextStyle.UNDERLINE]]`` to bold-and-underline
                just 1st place. Ranks beyond ``len(rank_styles)``, and any cell
                not reached by ``rank_styles`` at all, are left with no styles.
        """
        # Fill in default arguments that can't be default parameters
        if rank_styles is None:
            rank_styles = [TableData.TextStyle.BOLD, TableData.TextStyle.UNDERLINE]

        # Rank each column independently, cell by cell. Positions, not labels,
        # since the index may hold repeated labels (e.g. a multirow grouping column).
        n_rows, n_cols = self.df.shape
        for col_pos in range(n_cols):

            # Pull this column's raw values back out of the segments
            data = pd.Series([self.df.iat[row_pos, col_pos][0].raw_value for row_pos in range(n_rows)])
            groups: List[Set[int]] = TableData._rank_data_in_Series(data, higher_is_better)

            # Default every row in this column to unstyled, then apply rank_styles place by place
            style_by_pos: List[List[TableData.TextStyle]] = [[] for _ in range(n_rows)]
            for rank, group in enumerate(groups):
                if rank >= len(rank_styles):
                    break
                rank_style = rank_styles[rank]
                styles = rank_style if isinstance(rank_style, list) else [rank_style]
                for row_pos in group:
                    style_by_pos[row_pos] = styles

            # Update styles on each cell's existing segment, preserving its text/color
            for row_pos, styles in enumerate(style_by_pos):
                segment = self.df.iat[row_pos, col_pos][0]
                self.df.iat[row_pos, col_pos] = [replace(segment, styles=styles)]

    # =========================================================================
    # =========================== Color Functions =============================
    # =========================================================================

    @staticmethod
    def color_fn_NAVY_RED_missing_or_equal(equal_value: float = 0.0,
        style: TableData.TableStyleName = TableStyleName.GEORGIA_TECH) -> Callable[[float], Optional[str]]:
        """Build a color_fn: red for NaN (missing) or values equal to equal_value, navy otherwise."""
        resolved_style = TableData.get_table_style(style)

        def color_fn(value: float) -> Optional[str]:
            return resolved_style.TextFailureColor if math.isnan(value) or value == equal_value else resolved_style.TextColor
        return color_fn

    @staticmethod
    def color_fn_NAVY_RED_missing_or_above(threshold: float,
        style: TableData.TableStyleName = TableStyleName.GEORGIA_TECH) -> Callable[[float], Optional[str]]:
        """Build a color_fn: red for NaN (missing) or values above threshold, navy otherwise."""
        resolved_style = TableData.get_table_style(style)

        def color_fn(value: float) -> Optional[str]:
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
    # ============================ Export Methods ==============================
    # =========================================================================
    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escape characters with special meaning in LaTeX (e.g. ``&``, which would otherwise be read as a column separator)."""
        replacements = {
            '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
            '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
            '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
        }
        return ''.join(replacements.get(ch, ch) for ch in text)

    @staticmethod
    def _segments_to_latex(segments: List[TableData.FormattedTextSegment]) -> str:
        """Render one cell's TextSegments as LaTeX, applying bold/underline/color in that order."""
        parts: List[str] = []
        for seg in segments:
            text = TableData._escape_latex(seg.text)
            if TableData.TextStyle.UNDERLINE in seg.styles:
                text = f"\\underline{{{text}}}"
            if TableData.TextStyle.BOLD in seg.styles:
                text = f"\\textbf{{{text}}}"
            if seg.color is not None:
                text = f"\\textcolor[HTML]{{{seg.color.lstrip('#')}}}{{{text}}}"
            parts.append(text)
        return ''.join(parts)

    def to_latex(self, save_path: str, caption: str, label: str,
        column_format: Optional[str] = None, use_star_env: bool = True) -> None:
        """
        Render this table as LaTeX and save it to ``save_path``, ready to paste into Overleaf.

        Bold/underline/color styling on each ``FormattedTextSegment`` (as
        produced by ``highlight_best_and_worst_results_by_column``/``merge_TableData``) is
        preserved via ``\\textbf``, ``\\underline``, and
        ``\\textcolor[HTML]{...}``.

        The saved file requires ``\\usepackage{xcolor}`` in the document
        preamble (for ``\\textcolor[HTML]``).

        Args:
            save_path: Output ``.tex`` file path.
            caption: Table caption.
            label: Table ``\\label``.
            column_format: LaTeX tabular column spec. Defaults to
                ``|l|c|c|...|c|`` (one bar-separated centered column per data
                column).
            use_star_env: If True, use the two-column-spanning ``table*``
                environment; otherwise the single-column ``table`` environment.
        """

        # Fill in default arguments that can't be default parameters
        if column_format is None:
            column_format = "|l|" + "c|" * len(self.df.columns)

        # Set up the table/tabular preamble
        env: str = "table*" if use_star_env else "table"
        lines: List[str] = [
            f"\\begin{{{env}}}[ht]",
            "    \\centering",
            "    \\footnotesize",
            "    \\setlength{\\tabcolsep}{6pt}",
            "    \\renewcommand{\\arraystretch}{1.25}",
            "",
            f"    \\caption{{{caption}}}",
            f"    \\label{{{label}}}",
            "",
            f"    \\begin{{tabular}}{{{column_format}}}",
            "    \\hline",
        ]

        # Header row: the table's title labels the row-index column, plain column names follow
        header_cells: List[str] = [str(self.df.attrs.get("title", ""))] + [str(c) for c in self.df.columns]
        lines.append("    " + " & ".join(f"\\textbf{{{TableData._escape_latex(h)}}}" for h in header_cells) + r" \\")
        lines.append("    \\hline")

        # Data rows: row_name is plain text (escape only), cells hold FormattedTextSegments (escape + style)
        for row_name, row in self.df.iterrows():
            row_cells: List[str] = [TableData._escape_latex(str(row_name))]
            row_cells.extend(TableData._segments_to_latex(segments) for segments in row)
            lines.append("    " + " & ".join(row_cells) + r" \\")
        lines.append("    \\hline")

        lines.append("    \\end{tabular}")
        lines.append(f"\\end{{{env}}}")

        with open(save_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"\nLatex table saved to {save_path}")

    def render_onto_ax(self, fig: plt.Figure, ax: Axes, tbl_bbox: Optional[List[float]] = None,
        font_size: int = 11, data_font_size: Optional[int] = None,
        heavy_divider_before: Callable[[int], bool] = lambda _: False,
        style: TableData.TableStyleName = TableStyleName.GEORGIA_TECH) -> None:
        """
        Render this styled table onto ax, including its post-render decorations.

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
        resolved_style = TableData.get_table_style(style)
        df = self.df

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
        post_render_cells: List[Tuple[int, int, List[TableData.FormattedTextSegment]]] = []
        for i, (_, row) in enumerate(df.iterrows()):
            it = i + 1
            for j, segments in enumerate(row):
                segments: List[TableData.FormattedTextSegment]
                jt = j + 1

                # In the case of a single non-underlined segment, update it now
                if len(segments) == 1 and TableData.TextStyle.UNDERLINE not in segments[0].styles:
                    seg = segments[0]
                    cell = tbl[it, jt]
                    if seg.color is not None:
                        cell.get_text().set_color(seg.color)
                    if TableData.TextStyle.BOLD in seg.styles:
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
                fontweight = 'bold' if TableData.TextStyle.BOLD in seg.styles else 'normal'
                fontstyle = 'italic' if TableData.TextStyle.UNDERLINE in seg.styles else 'normal'
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
                                     fontweight='bold' if TableData.TextStyle.BOLD in seg.styles else 'normal',
                                     fontstyle='italic' if TableData.TextStyle.UNDERLINE in seg.styles else 'normal',
                                     ha='center', va='center')
                if TableData.TextStyle.UNDERLINE in seg.styles: # Remember to underline this after it is drawn later
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
    def to_pdf(tables: List[TableData], save_path: str, row_height: float = 2.4, h_pad: float = 1.2,
        heavy_divider_before: Callable[[int], bool] = lambda _: False,
        style: TableData.TableStyleName = TableStyleName.GEORGIA_TECH) -> None:
        """
        Render a list of styled tables on a single PDF.

        Args:
            tables: List of TableData, each whose underlying DataFrame's cells
                hold a ``List[TableData.FormattedTextSegment]`` (as produced by
                ``highlight_best_and_worst_results_by_column``/``merge_TableData``), with
                the table's title in ``df.attrs["title"]``.
            save_path: Output file path (PDF or PNG).
            row_height: Figure height per table in inches.
            h_pad: Vertical padding between subplots passed to tight_layout. The outer
                figure pad is fixed at 0.1 font-size units to minimise top/bottom margins.
            heavy_divider_before: callable(col_idx: int) -> bool; forwarded to
                ``render_onto_ax`` for every table (e.g. to set off a
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
        for ax, table in zip(axes, tables):
            table.render_onto_ax(fig, ax, heavy_divider_before=heavy_divider_before, style=style)

        # Save the figure
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"\nTables saved to {save_path}")

    # =========================================================================
    # ======================= Multi-TableData Methods =========================
    # =========================================================================

    @staticmethod
    def merge_TableData(*tables: TableData, separator: str = '/',
        style: TableData.TableStyleName = TableStyleName.GEORGIA_TECH) -> TableData:
        """
        Combine multiple identically-shaped TableData tables into one, joining each cell's segments with a separator.

        Typical use: call ``highlight_best_and_worst_results_by_column`` once
        per source table (e.g. one holding "successful" counts, one holding
        "total" counts, each ranked independently), then merge them into a
        single "successful/total" cell per position.

        Args:
            *tables: Two or more TableData instances sharing the same index
                and columns.
            separator: Text placed between each table's segments in a cell (e.g. "/").
            style: Named table style whose ``TextColor`` is used for the separator.

        Returns:
            A new TableData whose cells hold the combined TextSegment lists,
            with ``df.attrs["title"]`` copied from the first table.

        Raises:
            ValueError: If the tables don't all share the same index, columns, and title.
        """
        first_df = tables[0].df
        title = first_df.attrs.get("title", "")
        for table in tables[1:]:
            if not first_df.index.equals(table.df.index) or not first_df.columns.equals(table.df.columns):
                raise ValueError("All tables passed to merge_TableData must share the same index and columns.")
            if table.df.attrs.get("title", "") != title:
                raise ValueError("All tables passed to merge_TableData must share the same title.")

        separator_segment = TableData.FormattedTextSegment(separator, color=TableData.get_table_style(style).TextColor)

        # Positional (.iat), not label (.at): the shared index may hold repeated
        # labels (e.g. a multirow grouping column).
        n_rows, n_cols = first_df.shape
        merged_df = pd.DataFrame(index=first_df.index, columns=first_df.columns, dtype=object)
        for row_pos in range(n_rows):
            for col_pos in range(n_cols):
                combined: List[TableData.FormattedTextSegment] = []
                for i, table in enumerate(tables):
                    if i > 0:
                        combined.append(separator_segment)
                    combined.extend(table.df.iat[row_pos, col_pos])
                merged_df.iat[row_pos, col_pos] = combined
        merged_df.attrs["title"] = title

        return TableData.from_formatted_DataFrame(merged_df)

    def append_TableData(self, *others: TableData, axis: int = 0) -> TableData:
        """
        Append other TableData tables to this one along the given axis.

        Args:
            *others: One or more TableData instances to append after this one.
            axis: ``1`` to append additional columns (all tables must share
                the same index/rows, and their column labels must all be
                unique across tables); ``0`` to append additional rows (all
                tables must share the same columns, and their row labels must
                all be unique across tables). Use a ``pd.MultiIndex`` (e.g.
                ``(Sequence, Method)``) if you want a repeating outer label
                for multirow grouping in ``to_latex`` — the full tuple stays
                unique even though the outer level repeats.

        Returns:
            A new TableData holding the combined data, with ``df.attrs["title"]``
            copied from this table.

        Raises:
            ValueError: If ``axis`` isn't 0 or 1, the titles don't all match,
                the shared axis doesn't match across tables, or the appended
                axis's labels aren't unique across tables.
        """
        if axis not in (0, 1):
            raise ValueError(f"axis must be 0 or 1, got {axis!r}.")

        # Ensure all titles match
        all_tables: Tuple[TableData, ...] = (self,) + others
        title: str = self.df.attrs.get("title", "")
        for table in others:
            if table.df.attrs.get("title", "") != title:
                raise ValueError("All tables passed to append_TableData must share the same title.")

        # Ensure all tables share the same labels on the non-appended axis
        shared_axis_name: str = "rows (index)" if axis == 1 else "columns"
        shared_labels: List[pd.Index] = [table.df.index if axis == 1 else table.df.columns for table in all_tables]
        if any(not shared_labels[0].equals(labels) for labels in shared_labels[1:]):
            raise ValueError(f"All tables must share the same {shared_axis_name} to append along axis={axis}.")

        # Ensure the appended axis's labels are unique across all tables
        appended_axis_name: str = "Columns" if axis == 1 else "Row labels"
        appended_labels: List = [label for table in all_tables for label in (table.df.columns if axis == 1 else table.df.index)]
        if len(appended_labels) != len(set(appended_labels)):
            raise ValueError(f"{appended_axis_name} across all tables must be unique to append along axis={axis}.")

        # Append the tables
        combined_df: pd.DataFrame = pd.concat([table.df for table in all_tables], axis=axis)
        combined_df.attrs["title"] = title
        return TableData.from_formatted_DataFrame(combined_df)
