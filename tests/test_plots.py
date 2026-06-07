import numpy as np

from franka_llm_drawing.evaluation import ExecutionLogger, ExecutionLogRow, write_jade_plots


def test_write_jade_plots_adds_actual_xy_force_overlay(tmp_path) -> None:
    logger = ExecutionLogger()
    for index, force in enumerate([0.2, 1.0, 2.0]):
        logger.append(
            ExecutionLogRow(
                t=0.1 * index,
                stroke_id="stroke",
                action_type="draw_line",
                desired_position=np.array([0.1 * index, 0.0, 0.0]),
                actual_position=np.array([0.1 * index, 0.01 * index, 0.0]),
                commanded_position=np.array([0.1 * index, 0.0, 0.0]),
                q=np.zeros(7),
                qdot=np.zeros(7),
                sigma_min=0.1,
                condition_number=20.0 + index,
                manipulability=0.01,
                lambda_dls=0.02,
                speed_scale=1.0,
                status="OK",
                joint_limit_margin=0.5,
                tip_orientation_error_deg=1.0,
                desired_normal_force_n=1.0,
                measured_normal_force_n=force,
                filtered_normal_force_n=force,
                normal_force_error_n=1.0 - force,
                normal_force_offset_m=0.0,
                force_control_active=True,
                contact_active=True,
            )
        )

    paths = write_jade_plots(logger, tmp_path)

    overlay = tmp_path / "actual_xy_colored_by_force.svg"
    compensated = tmp_path / "planned_vs_actual_xy_offset_compensated.svg"
    draw_only = tmp_path / "planned_vs_actual_xy_draw_only.svg"
    assert overlay in paths
    assert overlay.exists()
    assert "measured normal force [N]" in overlay.read_text(encoding="utf-8")
    assert compensated in paths
    assert compensated.exists()
    assert "actual + mean offset" in compensated.read_text(encoding="utf-8")
    assert draw_only in paths
    assert draw_only.exists()
