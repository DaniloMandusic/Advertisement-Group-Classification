import matplotlib.pyplot as plt
import numpy as np

def plot_figure(df, columns, plot_fn, n_cols=3, figsize=(15, 5)):
    """
    Generic EDA plotting utility.

    Parameters:
    - df: DataFrame
    - columns: list of columns
    - plot_fn: function(ax, series, col_name)
    - n_cols: subplots per row
    """

    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize[0], figsize[1] * n_rows)
    )

    axes = axes.flatten() if len(columns) > 1 else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        plot_fn(ax, df[col], col)
        ax.set_title(col)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_figure_pairs(df, pairs, plot_fn, n_cols=3, figsize=(15, 5)):
    n_rows = (len(pairs) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize[0], figsize[1] * n_rows)
    )

    # 🔥 ALWAYS normalize axes to 1D array
    axes = np.array(axes).flatten()

    for i, (c1, c2) in enumerate(pairs):
        ax = axes[i]

        plot_fn(ax, df[c1], df[c2], c1, c2)

        ax.set_title(f"{c1} vs {c2}")

    # hide unused subplots safely
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()