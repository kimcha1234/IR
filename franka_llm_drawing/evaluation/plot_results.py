"""Optional plotting helpers for desired and actual drawing traces."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_xy_trace(
    p_des: np.ndarray,
    p_act: np.ndarray | None = None,
    out_path: str | Path | None = None,
) -> None:
    """Plot desired and optional actual XY paths.

    Matplotlib is imported lazily so offline tests do not require it.
    """

    import matplotlib.pyplot as plt

    des = np.asarray(p_des, dtype=float)
    if des.ndim != 2 or des.shape[1] != 3:
        raise ValueError("p_des must have shape (N, 3).")
    fig, ax = plt.subplots()
    ax.plot(des[:, 0], des[:, 1], label="desired")
    if p_act is not None:
        act = np.asarray(p_act, dtype=float)
        if act.shape != des.shape:
            raise ValueError("p_act must match p_des shape.")
        ax.plot(act[:, 0], act[:, 1], label="actual")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    fig.tight_layout()
    if out_path is None:
        plt.show()
    else:
        fig.savefig(out_path)
    plt.close(fig)
