# Single Photo Resection EOP Adjustment

This code implements a single-photo resection workflow to estimate and refine the exterior orientation parameters (EOPs) of one nadir-view image.

Using image observations and ground control points, the workflow:

- uses fixed initial IOP values from camera calibration,
- converts pixel coordinates into image coordinates,
- estimates initial EOPs from geometry or accepts supplied initial EOP values from GPS,
- formulates the collinearity equations, and
- refines the EOPs through an iterative Gauss-Markov least-squares adjustment.

## Input

The input file is:

- `spr_inputs.csv`

Expected columns:

- `GCP`
- `X(m)`
- `Y(m)`
- `Z(m)`
- `x(px)`
- `y(px)`

## Output

The script writes the following files to the `results/` directory:

- `initial_eop.csv`
- `final_eop.csv`
- `residuals.csv`
- `observation_vector.csv`

It also prints:

- initial EOP
- iteration count
- final EOP
- variance factor

## Initial EOP options

The script supports two initialization modes:

1. Estimate the initial EOP from the control point geometry
2. Use manually provided initial EOP values

This behavior is controlled by the `estimate_initial_eop` flag in the script.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt