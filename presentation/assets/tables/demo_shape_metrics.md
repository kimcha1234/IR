# Demo Shape Metrics

All metrics are computed only during the drawing-contact phase.

| Shape | XY RMSE (mm) | Max XY Error (mm) | Mean Force (N) | Max Force (N) | Contact Ratio (%) | Max Tip Ori. Error (deg) | Min Joint Margin (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Circle | 0.754 | 2.355 | 4.005 | 11.485 | 100.0 | 1.594 | 0.578 |
| Square | 0.868 | 2.119 | 3.828 | 10.196 | 100.0 | 1.569 | 0.605 |
| Triangle | 0.731 | 2.263 | 3.100 | 10.868 | 100.0 | 1.546 | 0.611 |
| Letter A | 1.197 | 2.303 | 4.514 | 8.977 | 100.0 | 1.527 | 0.614 |
| House | 0.809 | 2.128 | 3.631 | 11.622 | 100.0 | 1.596 | 0.575 |
| Star | 0.683 | 2.282 | 3.146 | 11.285 | 100.0 | 1.594 | 0.585 |

Key message for slides:
- Drawing XY tracking RMSE stayed around 0.7--1.2 mm for all six demo shapes.
- Drawing contact ratio was 100% for all six demo shapes.
- Tip orientation error stayed below about 1.6 deg during drawing.
- Force control is best described as contact maintenance and force monitoring, not perfect 1 N force tracking.
