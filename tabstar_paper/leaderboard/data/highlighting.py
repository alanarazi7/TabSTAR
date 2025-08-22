import pandas as pd
import streamlit as st
from typing import Optional, List

# TODO: this is chatGPT function, could be simplified
def st_dataframe_highlight_extremes(
    df: pd.DataFrame,
    highlight_subset: Optional[List[str]] = None,
    decimal_places: int = 4,
    worst_color: str = 'rgba(255,0,0,0.6)',
    best_color: str = 'rgba(0,255,0,0.6)',
    exclude_columns: Optional[List[str]] = None
):
    """
    Display a Streamlit DataFrame highlighting only the worst (min) and best (max)
    values in each row, using semi-transparent red and green respectively.

    Numeric columns are formatted to a fixed decimal precision, while integer columns
    (e.g., 'run_num') are displayed without decimals.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to style.
    highlight_subset : list of str, optional
        Columns to consider for highlighting. Defaults to all numeric columns
        excluding any in `exclude_columns`.
    decimal_places : int, optional
        Number of decimal places for float formatting.
    worst_color : str, optional
        RGBA color for worst values (default semi-transparent red).
    best_color : str, optional
        RGBA color for best values (default semi-transparent green).
    exclude_columns : list of str, optional
        Column names to always omit from highlighting (e.g. ['run_num']).

    Returns:
    --------
    None
        Renders the styled DataFrame in Streamlit.
    """
    # Determine numeric columns and split by type
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    int_cols = df.select_dtypes(include='integer').columns.tolist()
    float_cols = [c for c in numeric_cols if c not in int_cols]

    # Handle exclude_columns
    if exclude_columns is None:
        exclude_columns = []
    if highlight_subset is None:
        highlight_subset = [c for c in float_cols + int_cols if c not in exclude_columns]
    else:
        highlight_subset = [c for c in highlight_subset if c not in exclude_columns]

    # Build formatter mapping: ints no decimals, floats with fixed precision
    fmt_mapping = {
        **{c: '{:d}' for c in int_cols if c in numeric_cols},
        **{c: f'{{:.{decimal_places}f}}' for c in float_cols}
    }

    def highlight_row(row: pd.Series):
        vals = row[highlight_subset]
        if vals.empty:
            return [''] * len(row)
        min_val = vals.min()
        max_val = vals.max()
        styles = []
        for col, v in row.items():
            if col in highlight_subset and pd.notnull(v):
                if v == max_val:
                    styles.append(f"background-color: {best_color}")
                elif v == min_val:
                    styles.append(f"background-color: {worst_color}")
                else:
                    styles.append("")
            else:
                styles.append("")
        return styles

    styled = (
        df.style
          .format(fmt_mapping)
          .apply(highlight_row, axis=1)
    )

    st.dataframe(styled, use_container_width=True)