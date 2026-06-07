"""Dependency-light SVG plots for JADE reports."""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np

from franka_llm_drawing.evaluation.execution_logger import ExecutionLogger


def write_jade_plots(logger: ExecutionLogger, output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        out / "planned_vs_actual_xy.svg",
        out / "tracking_error.svg",
        out / "condition_number.svg",
        out / "adaptive_policy.svg",
    ]
    desired = logger.desired_positions()
    actual = logger.actual_positions()
    t = logger.times()
    _write_xy_plot(paths[0], desired[:, :2], actual[:, :2])
    draw_rows = [row for row in logger.rows if row.action_type in {"draw_line", "draw_line_to", "draw_arc"}]
    if len(draw_rows) >= 2:
        draw_desired = np.asarray([row.desired_position for row in draw_rows], dtype=float)
        draw_actual = np.asarray([row.actual_position for row in draw_rows], dtype=float)
        draw_path = out / "planned_vs_actual_xy_draw_only.svg"
        paths.append(draw_path)
        _write_xy_plot(draw_path, draw_desired[:, :2], draw_actual[:, :2])
        offset_path = out / "planned_vs_actual_xy_offset_compensated.svg"
        paths.append(offset_path)
        _write_xy_offset_compensated_plot(offset_path, draw_desired[:, :2], draw_actual[:, :2])
    _write_line_plot(paths[1], t, logger.error_norms(), "tracking error norm [m]")
    _write_line_plot(
        paths[2],
        t,
        np.asarray([row.condition_number for row in logger.rows], dtype=float),
        "condition number",
    )
    _write_multi_line_plot(
        paths[3],
        t,
        [
            (np.asarray([row.lambda_dls for row in logger.rows], dtype=float), "lambda DLS"),
            (np.asarray([row.speed_scale for row in logger.rows], dtype=float), "speed scale"),
        ],
    )
    orientation_errors = _optional_series([row.tip_orientation_error_deg for row in logger.rows])
    if orientation_errors is not None:
        path = out / "tip_orientation_error.svg"
        paths.append(path)
        _write_line_plot(path, t, orientation_errors, "tip orientation error [deg]")
    joint_margins = _optional_series([row.joint_limit_margin for row in logger.rows])
    if joint_margins is not None:
        path = out / "joint_limit_margin.svg"
        paths.append(path)
        _write_line_plot(path, t, joint_margins, "joint limit margin [rad]")
    measured_force = _optional_series([row.measured_normal_force_n for row in logger.rows])
    filtered_force = _optional_series([row.filtered_normal_force_n for row in logger.rows])
    desired_force = _optional_series([row.desired_normal_force_n for row in logger.rows])
    if measured_force is not None and filtered_force is not None and desired_force is not None:
        path = out / "normal_force.svg"
        paths.append(path)
        _write_multi_line_plot(
            path,
            t,
            [
                (desired_force, "desired normal force [N]"),
                (measured_force, "measured normal force [N]"),
                (filtered_force, "filtered normal force [N]"),
            ],
            ylabel="normal force [N]",
        )
        force_path = out / "actual_xy_colored_by_force.svg"
        paths.append(force_path)
        _write_xy_colored_by_scalar_plot(
            force_path,
            actual[:, :2],
            measured_force,
            xlabel="x [m]",
            ylabel="y [m]",
            color_label="measured normal force [N]",
        )
    force_offsets = _optional_series([row.normal_force_offset_m for row in logger.rows])
    if force_offsets is not None:
        path = out / "normal_force_offset.svg"
        paths.append(path)
        _write_line_plot(path, t, force_offsets, "normal force offset [m]")
    return paths


def _write_xy_plot(path: Path, desired_xy: np.ndarray, actual_xy: np.ndarray) -> None:
    series = [(desired_xy[:, 0], desired_xy[:, 1], "#1f77b4", "planned")]
    if actual_xy.size:
        series.append((actual_xy[:, 0], actual_xy[:, 1], "#d62728", "actual"))
    _write_svg(path, series, "x [m]", "y [m]", equal_aspect=True)


def _write_xy_offset_compensated_plot(path: Path, desired_xy: np.ndarray, actual_xy: np.ndarray) -> None:
    desired = np.asarray(desired_xy, dtype=float)
    actual = np.asarray(actual_xy, dtype=float)
    if desired.shape != actual.shape or desired.ndim != 2 or desired.shape[1] != 2:
        raise ValueError("desired_xy and actual_xy must both have shape (N, 2).")
    offset = np.mean(desired - actual, axis=0)
    compensated = actual + offset
    _write_svg(
        path,
        [
            (desired[:, 0], desired[:, 1], "#1f77b4", "planned"),
            (compensated[:, 0], compensated[:, 1], "#d62728", f"actual + mean offset ({offset[0]:+.4f}, {offset[1]:+.4f}) m"),
        ],
        "x [m]",
        "y [m]",
        equal_aspect=True,
    )


def _write_line_plot(path: Path, x: np.ndarray, y: np.ndarray, ylabel: str) -> None:
    _write_svg(path, [(x, y, "#1f77b4", ylabel)], "t [s]", ylabel, equal_aspect=False)


def _write_multi_line_plot(
    path: Path,
    x: np.ndarray,
    series: list[tuple[np.ndarray, str]],
    ylabel: str = "policy value",
) -> None:
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    svg_series = [(x, values, colors[i % len(colors)], label) for i, (values, label) in enumerate(series)]
    _write_svg(path, svg_series, "t [s]", ylabel, equal_aspect=False)


def _write_xy_colored_by_scalar_plot(
    path: Path,
    xy: np.ndarray,
    scalar: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    color_label: str,
) -> None:
    xy_arr = np.asarray(xy, dtype=float)
    scalar_arr = np.asarray(scalar, dtype=float)
    if xy_arr.ndim != 2 or xy_arr.shape[1] != 2:
        raise ValueError(f"xy must have shape (N, 2), got {xy_arr.shape}.")
    if scalar_arr.shape != (xy_arr.shape[0],):
        raise ValueError(f"scalar must have shape ({xy_arr.shape[0]},), got {scalar_arr.shape}.")
    width, height = 860, 560
    left, right, top, bottom = 82, 118, 46, 76
    x_min, x_max = _bounds(xy_arr[:, 0])
    y_min, y_max = _bounds(xy_arr[:, 1])
    span = max(x_max - x_min, y_max - y_min)
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    x_min, x_max = x_mid - 0.5 * span, x_mid + 0.5 * span
    y_min, y_max = y_mid - 0.5 * span, y_mid + 0.5 * span
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_ticks = _nice_ticks(x_min, x_max, 6)
    y_ticks = _nice_ticks(y_min, y_max, 6)
    x_step = _tick_step(x_ticks)
    y_step = _tick_step(y_ticks)
    f_min, f_max = _scalar_bounds(scalar_arr)

    def sx(x: np.ndarray) -> np.ndarray:
        return left + (x - x_min) / max(x_max - x_min, 1e-12) * plot_w

    def sy(y: np.ndarray) -> np.ndarray:
        return height - bottom - (y - y_min) / max(y_max - y_min, 1e-12) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfbfb" stroke="#333" stroke-width="1"/>',
        f'<text x="{width / 2}" y="{height - 20}" font-family="sans-serif" font-size="15" text-anchor="middle">{_esc(xlabel)}</text>',
        f'<text x="24" y="{top + plot_h / 2}" font-family="sans-serif" font-size="15" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})">{_esc(ylabel)}</text>',
        f'<text x="{left}" y="{top - 18}" font-family="sans-serif" font-size="12" fill="#444">{_esc(color_label)}: {_format_tick(f_min, f_max - f_min)}..{_format_tick(f_max, f_max - f_min)}</text>',
    ]
    for tick in x_ticks:
        x = float(sx(np.asarray([tick]))[0])
        elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#b8b8b8" stroke-width="0.8"/>')
        elements.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#333" stroke-width="1"/>')
        elements.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 23}" font-family="monospace" font-size="12" text-anchor="middle" fill="#333">{_format_tick(tick, x_step)}</text>'
        )
    for tick in y_ticks:
        y = float(sy(np.asarray([tick]))[0])
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#b8b8b8" stroke-width="0.8"/>')
        elements.append(f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#333" stroke-width="1"/>')
        elements.append(
            f'<text x="{left - 9}" y="{y + 4:.2f}" font-family="monospace" font-size="12" text-anchor="end" fill="#333">{_format_tick(tick, y_step)}</text>'
        )
    if xy_arr.shape[0] >= 2:
        x_screen = sx(xy_arr[:, 0])
        y_screen = sy(xy_arr[:, 1])
        for i in range(xy_arr.shape[0] - 1):
            value = 0.5 * (scalar_arr[i] + scalar_arr[i + 1])
            color = _scalar_color(value, f_min, f_max)
            elements.append(
                f'<line x1="{x_screen[i]:.2f}" y1="{y_screen[i]:.2f}" '
                f'x2="{x_screen[i + 1]:.2f}" y2="{y_screen[i + 1]:.2f}" '
                f'stroke="{color}" stroke-width="3.0" stroke-linecap="round"/>'
            )
    elements.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333" stroke-width="1"/>')
    _append_colorbar(elements, width - right + 36, top, 18, plot_h, f_min, f_max, color_label)
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_svg(
    path: Path,
    series: list[tuple[np.ndarray, np.ndarray, str, str]],
    xlabel: str,
    ylabel: str,
    *,
    equal_aspect: bool,
) -> None:
    width, height = 860, 560
    left, right, top, bottom = 82, 36, 46, 76
    xs = np.concatenate([np.asarray(item[0], dtype=float) for item in series])
    ys = np.concatenate([np.asarray(item[1], dtype=float) for item in series])
    x_min, x_max = _bounds(xs)
    y_min, y_max = _bounds(ys)
    if equal_aspect:
        span = max(x_max - x_min, y_max - y_min)
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        x_min, x_max = x_mid - 0.5 * span, x_mid + 0.5 * span
        y_min, y_max = y_mid - 0.5 * span, y_mid + 0.5 * span
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_ticks = _nice_ticks(x_min, x_max, 6)
    y_ticks = _nice_ticks(y_min, y_max, 6)
    x_step = _tick_step(x_ticks)
    y_step = _tick_step(y_ticks)

    def sx(x: np.ndarray) -> np.ndarray:
        return left + (x - x_min) / max(x_max - x_min, 1e-12) * plot_w

    def sy(y: np.ndarray) -> np.ndarray:
        return height - bottom - (y - y_min) / max(y_max - y_min, 1e-12) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfbfb" stroke="#333" stroke-width="1"/>',
        f'<text x="{width / 2}" y="{height - 20}" font-family="sans-serif" font-size="15" text-anchor="middle">{_esc(xlabel)}</text>',
        f'<text x="24" y="{top + plot_h / 2}" font-family="sans-serif" font-size="15" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})">{_esc(ylabel)}</text>',
        f'<text x="{left}" y="{top - 18}" font-family="sans-serif" font-size="12" fill="#444">{_esc(xlabel)}: {_format_tick(x_min, x_step)}..{_format_tick(x_max, x_step)}; {_esc(ylabel)}: {_format_tick(y_min, y_step)}..{_format_tick(y_max, y_step)}</text>',
    ]
    for tick in x_ticks:
        x = float(sx(np.asarray([tick]))[0])
        stroke = "#b8b8b8" if not math.isclose(tick, 0.0, abs_tol=1e-12) else "#888"
        elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="{stroke}" stroke-width="0.8"/>')
        elements.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#333" stroke-width="1"/>')
        elements.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 23}" font-family="monospace" font-size="12" text-anchor="middle" fill="#333">{_format_tick(tick, x_step)}</text>'
        )
    for tick in y_ticks:
        y = float(sy(np.asarray([tick]))[0])
        stroke = "#b8b8b8" if not math.isclose(tick, 0.0, abs_tol=1e-12) else "#888"
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="{stroke}" stroke-width="0.8"/>')
        elements.append(f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#333" stroke-width="1"/>')
        elements.append(
            f'<text x="{left - 9}" y="{y + 4:.2f}" font-family="monospace" font-size="12" text-anchor="end" fill="#333">{_format_tick(tick, y_step)}</text>'
        )
    elements.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333" stroke-width="1"/>')
    for index, (x_values, y_values, color, label) in enumerate(series):
        points = " ".join(
            f"{x:.2f},{y:.2f}" for x, y in zip(sx(np.asarray(x_values)), sy(np.asarray(y_values)))
        )
        elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        legend_x = width - right - 145
        legend_y = top + 18 + 20 * index
        elements.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 28}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>')
        elements.append(
            f'<text x="{legend_x + 36}" y="{legend_y}" '
            f'font-family="sans-serif" font-size="13" fill="{color}">{_esc(label)}</text>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _bounds(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if abs(v_max - v_min) < 1e-12:
        pad = max(abs(v_min) * 0.1, 1e-3)
    else:
        pad = 0.08 * (v_max - v_min)
    return v_min - pad, v_max + pad


def _scalar_bounds(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    v_min = float(np.min(finite))
    v_max = float(np.max(finite))
    if abs(v_max - v_min) < 1e-12:
        pad = max(abs(v_min) * 0.1, 1e-3)
        return v_min - pad, v_max + pad
    return v_min, v_max


def _scalar_color(value: float, v_min: float, v_max: float) -> str:
    ratio = (float(value) - float(v_min)) / max(float(v_max) - float(v_min), 1e-12)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if ratio < 0.5:
        local = ratio / 0.5
        c0 = np.asarray([44, 123, 182], dtype=float)
        c1 = np.asarray([255, 255, 191], dtype=float)
    else:
        local = (ratio - 0.5) / 0.5
        c0 = np.asarray([255, 255, 191], dtype=float)
        c1 = np.asarray([215, 25, 28], dtype=float)
    rgb = np.round((1.0 - local) * c0 + local * c1).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _append_colorbar(
    elements: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    v_min: float,
    v_max: float,
    label: str,
) -> None:
    steps = 40
    for i in range(steps):
        ratio0 = i / steps
        ratio1 = (i + 1) / steps
        value = v_min + (1.0 - ratio0) * (v_max - v_min)
        color = _scalar_color(value, v_min, v_max)
        y0 = y + ratio0 * height
        h = max((ratio1 - ratio0) * height, 1.0)
        elements.append(f'<rect x="{x:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{h:.2f}" fill="{color}" stroke="none"/>')
    elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="none" stroke="#333" stroke-width="1"/>')
    elements.append(
        f'<text x="{x + width + 7:.2f}" y="{y + 4:.2f}" font-family="monospace" font-size="11" fill="#333">{_format_tick(v_max, v_max - v_min)}</text>'
    )
    elements.append(
        f'<text x="{x + width + 7:.2f}" y="{y + height:.2f}" font-family="monospace" font-size="11" fill="#333">{_format_tick(v_min, v_max - v_min)}</text>'
    )
    elements.append(
        f'<text x="{x - 8:.2f}" y="{y + height + 32:.2f}" font-family="sans-serif" font-size="11" fill="#333" transform="rotate(-90 {x - 8:.2f} {y + height + 32:.2f})">{_esc(label)}</text>'
    )


def _nice_ticks(v_min: float, v_max: float, max_ticks: int) -> list[float]:
    span = float(v_max - v_min)
    if not np.isfinite(span) or span <= 0.0:
        return [float(v_min)]
    raw_step = span / max(max_ticks - 1, 1)
    base = 10.0 ** math.floor(math.log10(raw_step))
    candidates = [1.0, 2.0, 2.5, 5.0, 10.0]
    step = candidates[-1] * base
    for candidate in candidates:
        step_candidate = candidate * base
        if raw_step <= step_candidate:
            step = step_candidate
            break
    first = math.ceil(v_min / step) * step
    last = math.floor(v_max / step) * step
    ticks = []
    value = first
    while value <= last + 0.5 * step:
        ticks.append(0.0 if math.isclose(value, 0.0, abs_tol=step * 1e-9) else float(value))
        value += step
    if not ticks:
        return [float(v_min), float(v_max)]
    return ticks


def _tick_step(ticks: list[float]) -> float:
    if len(ticks) < 2:
        return 1.0
    return abs(float(ticks[1] - ticks[0]))


def _format_tick(value: float, step: float) -> str:
    if not np.isfinite(value):
        return str(value)
    step_abs = abs(float(step))
    if step_abs == 0.0:
        decimals = 3
    elif step_abs >= 1.0:
        decimals = 0
    else:
        decimals = min(5, max(1, int(math.ceil(-math.log10(step_abs))) + 1))
    if abs(value) >= 10000 or (0 < abs(value) < 1e-4):
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


def _esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def _optional_series(values: list[float | None]) -> np.ndarray | None:
    if not values or any(value is None for value in values):
        return None
    series = np.asarray(values, dtype=float)
    if not np.isfinite(series).all():
        return None
    return series
