# Presentation Assets

This folder collects tables, graphs, one environment screenshot, and representative Isaac Sim demo videos for the final presentation.

## Recommended Slide Usage

- Slide 3 System Architecture: `tables/controller_architecture_summary.md`
- Slide 4 LLM Planner & Unit Actions: `tables/demo_commands.md`, `demo_plans_and_reports/plans/*.json`
- Slide 6 Robot Modeling and Simulation Environment: `entire_robot.png`
- Slide 7 Primitive Actions to Robot Motion: `recommended_for_slides/slide07_star_planned_vs_actual_draw_only.svg`
- Slide 9 Jacobian-Based Motion Control: `recommended_for_slides/slide09_star_condition_number.svg`, `slide09_star_adaptive_dls.svg`, `slide09_star_joint_limit_margin.svg`
- Slide 10 Hybrid Position-Force Control: `recommended_for_slides/slide10_star_normal_force.svg`, `slide10_star_force_colored_trajectory.svg`
- Slide 11 Results: `tables/demo_shape_metrics.md`, `summary_graphs/*.png`, `video/*.mp4`

## Folder Contents

- `tables/`: CSV, Markdown, and LaTeX table snippets.
- `graphs_by_shape/`: all final SVG graphs, summary JSON, and execution logs for each demo shape.
- `recommended_for_slides/`: a smaller curated set of graphs copied with slide-oriented names.
- `summary_graphs/`: aggregate bar charts generated from the six demo summaries.
- `demo_plans_and_reports/`: natural-language demo plan JSON files and validation reports.
- `video/`: representative Isaac Sim demo recordings. These are not intended to cover every shape.
- `entire_robot.png`: Isaac Sim environment screenshot showing the Franka, pen, table, and paper.

## Notes for Presentation

- Position tracking should be presented using drawing-phase metrics.
- Contact force should be presented as contact maintenance and force monitoring, not as perfect 1 N force tracking.
- Top-view drawing behavior is already documented by the XY trajectory graphs, so no separate top-view screenshot is required.
- Demo videos are representative examples; the quantitative tables and graphs cover all six shapes.
- Isaac backend commands only `panda_joint1` through `panda_joint7`.
