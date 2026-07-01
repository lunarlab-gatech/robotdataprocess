from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import pandas as pd
from typing import Callable, List, Tuple, Optional

HEADER_COLOR = '#B5A060'
ROW_COLORS   = ['#EDEADE', '#D8D3C3']
TEXT_COLOR   = '#1A3055'
ZERO_COLOR   = '#CC2222'
FONT_SIZE    = 11


def _is_zero(val_str: str) -> bool:
    try:
        return float(val_str.strip().rstrip('%')) == 0.0
    except ValueError:
        return False


def _col_rank_groups(rank_df: pd.DataFrame, col: str) -> List[set]:
    """Return row-name groups ordered best-to-worst, with ties in the same group."""
    vals = rank_df[col].astype(float)
    groups = []
    for v in sorted(vals.unique(), reverse=True):
        groups.append(set(vals[vals == v].index))
    return groups


def _render_table(ax, df: pd.DataFrame, title: str,
                  rank_dfs: Optional[List[pd.DataFrame]] = None,
                  cell_is_red: Callable[[str], bool] = _is_zero,
                  tbl_bbox: Optional[List[float]] = None,
                  font_size: Optional[int] = None,
                  data_font_size: Optional[int] = None):
    """Render one styled table onto ax.

    rank_dfs: list of DataFrames with numeric values for per-column highlighting.
      Single rank_df: whole-cell bold/italic+underline.
      Multiple rank_dfs (split mode): each rank_df controls one part of an 'X/Y' cell —
        only the winning part is bolded/underlined, not the whole cell.
    cell_is_red: callable(val_str) -> bool; red cells skip bold/underline.
      Defaults to _is_zero (red when value == 0).

    Returns (underline_cells, split_cells, col_divider_cells) for post-render drawing.
    """
    ax.axis('off')
    col_labels = [title] + list(df.columns)
    cell_text  = [[str(idx)] + [str(v) for v in row] for idx, row in df.iterrows()]
    n_rows, n_cols = len(cell_text), len(col_labels)

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   bbox=tbl_bbox if tbl_bbox is not None else [0, 0, 1, 1],
                   cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size if font_size is not None else FONT_SIZE)
    tbl.auto_set_column_width(col=list(range(n_cols)))
    if data_font_size is not None:
        for i in range(1, n_rows + 1):
            for j in range(1, n_cols):
                tbl[i, j].get_text().set_fontsize(data_font_size)

    # Halve the first (method name) column width
    for row_i in range(n_rows + 1):
        cell = tbl[row_i, 0]
        cell.set_width(cell.get_width() * 0.5)

    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_edgecolor('none')
        cell.get_text().set_color('white')
        cell.get_text().set_fontweight('bold')

    for i in range(n_rows):
        bg = ROW_COLORS[i % 2]
        for j in range(n_cols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor('none')
            val_str = cell_text[i][j]
            color = ZERO_COLOR if (j > 0 and cell_is_red(val_str)) else TEXT_COLOR
            cell.get_text().set_color(color)
            if j == 0:
                cell.get_text().set_ha('left')

    col_divider_cells = [(tbl, n_rows, j) for j in range(1, n_cols)]

    underline_cells = []
    split_cells = []

    if rank_dfs:
        row_order = list(df.index)
        is_split = len(rank_dfs) > 1

        for j, col in enumerate(df.columns):
            if is_split:
                part_bold = []
                part_ul   = []
                for rank_df in rank_dfs:
                    groups = _col_rank_groups(rank_df, col)
                    bold = groups[0] if len(groups) >= 1 else set()
                    ul   = (groups[1] if len(groups) >= 2 else set()) - bold
                    part_bold.append(bold)
                    part_ul.append(ul)

                for i, row_name in enumerate(row_order):
                    ri = i + 1
                    cell_val = str(df.loc[row_name, col])
                    x_str, y_str = cell_val.split('/')

                    x_bold = (row_name in part_bold[0]) and not cell_is_red(x_str)
                    x_ul   = (row_name in part_ul[0])   and not cell_is_red(x_str)
                    y_bold = (row_name in part_bold[1]) and not cell_is_red(y_str)
                    y_ul   = (row_name in part_ul[1])   and not cell_is_red(y_str)

                    if x_bold or x_ul or y_bold or y_ul:
                        tbl[ri, j + 1].get_text().set_text('')
                        x_color = ZERO_COLOR if cell_is_red(x_str) else TEXT_COLOR
                        y_color = ZERO_COLOR if cell_is_red(y_str) else TEXT_COLOR
                        parts = [
                            (x_str, 'bold' if x_bold else 'normal', 'italic' if x_ul else 'normal', x_ul, x_color),
                            ('/',   'normal', 'normal', False, TEXT_COLOR),
                            (y_str, 'bold' if y_bold else 'normal', 'italic' if y_ul else 'normal', y_ul, y_color),
                        ]
                        split_cells.append((tbl, ri, j + 1, parts))
                    elif cell_is_red(x_str) or cell_is_red(y_str):
                        tbl[ri, j + 1].get_text().set_text('')
                        x_color = ZERO_COLOR if cell_is_red(x_str) else TEXT_COLOR
                        y_color = ZERO_COLOR if cell_is_red(y_str) else TEXT_COLOR
                        parts = [
                            (x_str, 'normal', 'normal', False, x_color),
                            ('/',   'normal', 'normal', False, TEXT_COLOR),
                            (y_str, 'normal', 'normal', False, y_color),
                        ]
                        split_cells.append((tbl, ri, j + 1, parts))

            else:
                groups = _col_rank_groups(rank_dfs[0], col)
                bold_rows = groups[0] if len(groups) >= 1 else set()
                ul_rows   = (groups[1] if len(groups) >= 2 else set()) - bold_rows

                for row_name in bold_rows:
                    ri = row_order.index(row_name) + 1
                    if not cell_is_red(str(df.loc[row_name, col])):
                        tbl[ri, j + 1].get_text().set_fontweight('bold')
                for row_name in ul_rows:
                    ri = row_order.index(row_name) + 1
                    if not cell_is_red(str(df.loc[row_name, col])):
                        tbl[ri, j + 1].get_text().set_fontstyle('italic')
                        underline_cells.append((tbl, ri, j + 1))

    return underline_cells, split_cells, col_divider_cells


def _apply_post_render(fig, underline_cells: list, split_cells: list,
                       col_divider_cells: list) -> None:
    """Apply post-render decorations (split-cell text, column dividers, underlines).

    Must be called after all tables have been rendered and the figure layout is
    final (e.g. after ``tight_layout``). Performs two ``canvas.draw()`` passes:
    the first to obtain renderer positions, the second to draw underlines on
    split-cell parts.

    Args:
        fig: The matplotlib figure containing all rendered tables.
        underline_cells: List of ``(tbl, row_idx, col_idx)`` tuples for
            whole-cell italic+underline entries (returned by ``_render_table``).
        split_cells: List of ``(tbl, row_idx, col_idx, parts)`` tuples for
            X/Y cells that need per-part styling (returned by ``_render_table``).
        col_divider_cells: List of ``(tbl, bottom_row_idx, col_idx)`` tuples
            marking where white vertical column dividers should be drawn
            (returned by ``_render_table``).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    split_underline_texts = []

    for tbl, ri, ci, parts in split_cells:
        cell = tbl[ri, ci]
        bb = cell.get_window_extent(renderer=renderer)

        widths = []
        for text_str, fontweight, fontstyle, *_ in parts:
            fp = FontProperties(size=FONT_SIZE, weight=fontweight, style=fontstyle)
            w, _, _ = renderer.get_text_width_height_descent(text_str, fp, ismath=False)
            widths.append(w)

        cx = (bb.x0 + bb.x1) / 2
        cy = (bb.y0 + bb.y1) / 2
        x_cursor = cx - sum(widths) / 2

        for k, (text_str, fontweight, fontstyle, is_ul, color) in enumerate(parts):
            fig_x, fig_y = fig.transFigure.inverted().transform(
                (x_cursor + widths[k] / 2, cy))
            txt = fig.text(fig_x, fig_y, text_str,
                           fontsize=FONT_SIZE, color=color,
                           fontweight=fontweight, fontstyle=fontstyle,
                           ha='center', va='center')
            if is_ul:
                split_underline_texts.append(txt)
            x_cursor += widths[k]

    for tbl, ri, ci in col_divider_cells:
        cell = tbl[ri, ci]
        bb = cell.get_window_extent(renderer=renderer)
        fig_x, _ = fig.transFigure.inverted().transform((bb.x0, 0))
        header_bb = tbl[0, ci].get_window_extent(renderer=renderer)
        bottom_bb = tbl[ri, ci].get_window_extent(renderer=renderer)
        fig_y0 = fig.transFigure.inverted().transform((0, bottom_bb.y0))[1]
        fig_y1 = fig.transFigure.inverted().transform((0, header_bb.y1))[1]
        fig.add_artist(mlines.Line2D(
            [fig_x, fig_x], [fig_y0, fig_y1],
            transform=fig.transFigure, color='white',
            linewidth=1.0, clip_on=False
        ))

    for tbl, ri, ci in underline_cells:
        txt = tbl[ri, ci].get_text()
        bb = txt.get_window_extent(renderer=renderer)
        fig_bb = bb.transformed(fig.transFigure.inverted())
        y = fig_bb.y0
        fig.add_artist(mlines.Line2D(
            [fig_bb.x0, fig_bb.x1], [y, y],
            transform=fig.transFigure, color=TEXT_COLOR,
            linewidth=0.8, clip_on=False
        ))

    if split_underline_texts:
        fig.canvas.draw()
        for txt in split_underline_texts:
            bb = txt.get_window_extent(renderer=renderer)
            fig_bb = bb.transformed(fig.transFigure.inverted())
            y = fig_bb.y0
            fig.add_artist(mlines.Line2D(
                [fig_bb.x0, fig_bb.x1], [y, y],
                transform=fig.transFigure, color=TEXT_COLOR,
                linewidth=0.8, clip_on=False
            ))


def render_tables_onto_axes(
    fig,
    table_specs: List[Tuple],
    cell_is_red: Callable[[str], bool] = _is_zero,
) -> None:
    """Render styled tables into existing axes and apply post-render decorations.

    The figure layout must be finalized (e.g. via ``tight_layout`` or fixed
    ``GridSpec`` margins) before calling this, as post-render drawing uses
    renderer-level bounding boxes.

    Args:
        fig: The matplotlib figure that owns all provided axes.
        table_specs: List of ``(ax, title, df, rank_dfs)`` tuples.
            Optional 5th element overrides ``cell_is_red`` for that table.
            Optional 6th element is a ``[x0, y0, width, height]`` bbox (axes
            coordinates) passed to ``ax.table``; controls how much of the axes
            the table fills.  Default is ``[0, 0, 1, 1]`` (full axes).
        cell_is_red: callable(val_str) -> bool; red cells skip bold/underline.
            Defaults to ``_is_zero``.
    """
    all_ul, all_sp, all_cd = [], [], []
    for spec in table_specs:
        ax, title, df, rank_dfs = spec[:4]
        tbl_cell_is_red  = spec[4] if len(spec) > 4 and spec[4] is not None else cell_is_red
        tbl_bbox         = spec[5] if len(spec) > 5 else None
        tbl_font_size    = spec[6] if len(spec) > 6 else None
        tbl_data_font    = spec[7] if len(spec) > 7 else None
        ul, sp, cd = _render_table(ax, df, title, rank_dfs, tbl_cell_is_red, tbl_bbox,
                                   tbl_font_size, tbl_data_font)
        all_ul += ul
        all_sp += sp
        all_cd += cd
    _apply_post_render(fig, all_ul, all_sp, all_cd)


def save_styled_tables(
    dfs: List[Tuple[str, pd.DataFrame, Optional[List[pd.DataFrame]]]],
    save_path: str,
    row_height: float = 2.4,
    h_pad: float = 1.2,
    cell_is_red: Callable[[str], bool] = _is_zero,
) -> None:
    """Render a list of styled tables and save them to a PDF.

    Args:
        dfs: List of (title, display_df, rank_dfs) tuples. rank_dfs may be None
            (no highlighting), a single-element list (whole-cell bold/underline),
            or a two-element list (split 'X/Y' cell mode).
        save_path: Output file path (PDF or PNG).
        row_height: Figure height per table in inches.
        h_pad: Vertical padding between subplots passed to tight_layout. The outer
            figure pad is fixed at 0.1 font-size units to minimise top/bottom margins.
        cell_is_red: callable(val_str) -> bool; matching cells are shown in red
            and excluded from bold/underline. Defaults to _is_zero (red when == 0).
    """
    fig, axes = plt.subplots(len(dfs), 1, figsize=(12, row_height * len(dfs)))
    if len(dfs) == 1:
        axes = [axes]
    for ax in axes:
        ax.axis('off')
    fig.tight_layout(pad=0.0, h_pad=h_pad)
    render_tables_onto_axes(fig, [(ax, title, df, rank_dfs)
                                  for ax, (title, df, rank_dfs) in zip(axes, dfs)],
                            cell_is_red)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\nTables saved to {save_path}")
