"""
Single photo resection with EOP refinement.

Author: Hyunjoo Ho
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import fsolve


class SinglePhotoResectionEOP:
    """Estimate and refine EOPs for a single image using fixed IOPs."""

    def __init__(self):
        # Image size in pixels
        self.img_w = 7952
        self.img_h = 5304

        # Pixel size in meters
        self.one_px = 4.4e-6

        # Fixed interior orientation parameters [xp, yp, c] in meters
        self.init_iop = np.array(
            [-0.094217e-3, -0.033517e-3, 32.784e-3],
            dtype=float,
        )

        # Expected input columns
        self.required_columns = ["GCP", "X(m)", "Y(m)", "Z(m)", "x(px)", "y(px)"]

    def load_inputs(self, csv_path: str | Path) -> pd.DataFrame:
        """Load SPR input data from CSV and convert image coordinates to meters."""
        df = pd.read_csv(csv_path).copy()

        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.dropna(subset=self.required_columns).copy()

        numeric_cols = ["X(m)", "Y(m)", "Z(m)", "x(px)", "y(px)"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_cols).copy()

        # Convert pixel coordinates to centered metric image coordinates
        df["x_m"] = (df["x(px)"] - self.img_w / 2.0) * self.one_px
        df["y_m"] = (self.img_h / 2.0 - df["y(px)"]) * self.one_px

        return df.reset_index(drop=True)
    

    def load_initial_eop(self, csv_path: str | Path) -> np.ndarray:
        """Load initial EOP values from a CSV file."""
        df = pd.read_csv(csv_path).copy()

        required_cols = ["Xo", "Yo", "Zo", "omega_deg", "phi_deg", "kappa_deg"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in initial EOP file: {missing}")

        if len(df) != 1:
            raise ValueError("Initial EOP file must contain exactly one row.")

        row = df.iloc[0]

        init_eop = np.array(
            [
                float(row["Xo"]),
                float(row["Yo"]),
                float(row["Zo"]),
                np.deg2rad(float(row["omega_deg"])),
                np.deg2rad(float(row["phi_deg"])),
                np.deg2rad(float(row["kappa_deg"])),
            ],
            dtype=float,
        )

        return init_eop

    def solve_height(self, XA, YA, ZA, xa, ya, XB, YB, ZB, xb, yb):
        """Solve the height equation used in the initial EOP estimation."""
        c = self.init_iop[2]

        def height_eq(H):
            ab_sq_ground = (XB - XA) ** 2 + (YB - YA) ** 2
            ab_sq_image = (
                (xb / c * (H - ZB) - xa / c * (H - ZA)) ** 2
                + (yb / c * (H - ZB) - ya / c * (H - ZA)) ** 2
            )
            return ab_sq_image - ab_sq_ground

        guesses = [100.0, 200.0, 500.0, 1000.0, ZA + 100.0, ZB + 100.0]
        roots = []

        for guess in guesses:
            try:
                root = float(fsolve(height_eq, guess)[0])
                if np.isfinite(root):
                    roots.append(root)
            except Exception:
                continue

        if not roots:
            raise RuntimeError("Failed to solve initial height.")

        # Use the largest valid root as a practical replacement
        # for the symbolic root selection in MATLAB
        roots = np.unique(np.round(np.array(roots), decimals=8))
        return float(np.max(roots))

    def estimate_initial_eop(self, df: pd.DataFrame) -> np.ndarray:
        """Estimate initial EOP from ground control geometry."""
        n_obs = len(df)
        if n_obs < 2:
            raise ValueError("At least 2 observations are required to estimate initial EOP.")

        c = self.init_iop[2]

        A1_rows = []
        Y1_rows = []
        H1 = None

        for k in range(n_obs):
            pt_a = df.iloc[k]
            pt_b = df.iloc[k + 1] if k < n_obs - 1 else df.iloc[k - 1]

            XA = float(pt_a["X(m)"])
            YA = float(pt_a["Y(m)"])
            ZA = float(pt_a["Z(m)"])
            xa = float(pt_a["x_m"])
            ya = float(pt_a["y_m"])

            XB = float(pt_b["X(m)"])
            YB = float(pt_b["Y(m)"])
            ZB = float(pt_b["Z(m)"])
            xb = float(pt_b["x_m"])
            yb = float(pt_b["y_m"])

            H1 = self.solve_height(XA, YA, ZA, xa, ya, XB, YB, ZB, xb, yb)

            Xarb = xa * (H1 - ZA) / c
            Yarb = ya * (H1 - ZA) / c

            A1_rows.append([Xarb, -Yarb, 1.0, 0.0])
            A1_rows.append([Yarb,  Xarb, 0.0, 1.0])

            Y1_rows.append(XA)
            Y1_rows.append(YA)

        A1 = np.array(A1_rows, dtype=float)
        Y1 = np.array(Y1_rows, dtype=float)

        dX1, *_ = np.linalg.lstsq(A1, Y1, rcond=None)

        init_kappa = np.arctan(dX1[1] / dX1[0])
        if dX1[0] < 0:
            if dX1[1] < 0:
                init_kappa = init_kappa - np.pi
            else:
                init_kappa = init_kappa + np.pi

        init_eop = np.array([dX1[2], dX1[3], H1, 0.0, 0.0, init_kappa], dtype=float)
        return init_eop

    def project_point(self, eop: np.ndarray, X: float, Y: float, Z: float) -> np.ndarray:
        """Project a 3D object point to image coordinates using collinearity equations."""
        Xo, Yo, Zo, omg, phi, kap = eop
        xp, yp, c = self.init_iop

        r11 = np.cos(phi) * np.cos(kap)
        r12 = -np.cos(phi) * np.sin(kap)
        r13 = np.sin(phi)

        r21 = np.cos(omg) * np.sin(kap) + np.sin(omg) * np.sin(phi) * np.cos(kap)
        r22 = np.cos(omg) * np.cos(kap) - np.sin(omg) * np.sin(phi) * np.sin(kap)
        r23 = -np.sin(omg) * np.cos(phi)

        r31 = np.sin(omg) * np.sin(kap) - np.cos(omg) * np.sin(phi) * np.cos(kap)
        r32 = np.sin(omg) * np.cos(kap) + np.cos(omg) * np.sin(phi) * np.sin(kap)
        r33 = np.cos(omg) * np.cos(phi)

        dX = X - Xo
        dY = Y - Yo
        dZ = Z - Zo

        Nx = r11 * dX + r21 * dY + r31 * dZ
        Ny = r12 * dX + r22 * dY + r32 * dZ
        De = r13 * dX + r23 * dY + r33 * dZ

        x = xp - c * Nx / De
        y = yp - c * Ny / De

        return np.array([x, y], dtype=float)

    def numerical_jacobian(self, eop: np.ndarray, X: float, Y: float, Z: float):
        """Compute the numerical Jacobian with respect to EOP parameters only."""
        f0 = self.project_point(eop, X, Y, Z)
        jac = np.zeros((2, 6), dtype=float)

        steps = np.array([1e-4, 1e-4, 1e-4, 1e-8, 1e-8, 1e-8], dtype=float)

        for i in range(6):
            trial = eop.copy()
            trial[i] += steps[i]
            f1 = self.project_point(trial, X, Y, Z)
            jac[:, i] = (f1 - f0) / steps[i]

        return jac, f0

    def run_least_squares(self, df: pd.DataFrame, init_eop: np.ndarray, max_iter: int = 100):
        """Run iterative EOP-only least-squares refinement."""
        current_eop = init_eop.copy()
        lim = 1e-6 * np.ones(6, dtype=float)

        iteration_count = 1

        while True:
            n_obs = len(df)
            A2 = np.zeros((n_obs * 2, 6), dtype=float)
            Y2 = []

            for k, row in df.iterrows():
                X = float(row["X(m)"])
                Y = float(row["Y(m)"])
                Z = float(row["Z(m)"])

                obs_x = float(row["x_m"])
                obs_y = float(row["y_m"])

                jac, pred = self.numerical_jacobian(current_eop, X, Y, Z)

                row1 = 2 * k
                row2 = 2 * k + 2
                A2[row1:row2, :] = jac

                Y2.append(obs_x - pred[0])
                Y2.append(obs_y - pred[1])

            Y2 = np.array(Y2, dtype=float)

            dX, *_ = np.linalg.lstsq(A2, Y2, rcond=None)

            if np.all(np.abs(dX) < lim):
                break

            current_eop = current_eop + dX
            iteration_count += 1

            if iteration_count > max_iter:
                print("Warning: maximum iteration count reached before convergence.")
                break

        e = Y2 - A2 @ dX
        dof = len(Y2) - 6
        var_cov = float((e.T @ e) / dof) if dof > 0 else np.nan

        return current_eop, iteration_count, e, var_cov, Y2

    @staticmethod
    def eop_to_table(eop: np.ndarray) -> pd.DataFrame:
        """Convert EOP values to a display table with angles in degrees."""
        return pd.DataFrame(
            {
                "Xo": [eop[0]],
                "Yo": [eop[1]],
                "Zo": [eop[2]],
                "omega_deg": [np.rad2deg(eop[3])],
                "phi_deg": [np.rad2deg(eop[4])],
                "kappa_deg": [np.rad2deg(eop[5])],
            }
        )


def main():
    """Run the complete single-photo resection workflow."""
    csv_path = Path("./data/spr_gcp_inputs.csv")
    initial_eop_file = Path("./data/spr_eop_inputs.csv")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    engine = SinglePhotoResectionEOP()
    df = engine.load_inputs(csv_path)

    init_eop_mode = "estimate"
    # Options:
    # "estimate" -> estimate initial EOP from geometry
    # "file"     -> load initial EOP from a CSV file

    if init_eop_mode == "estimate":
        init_eop = engine.estimate_initial_eop(df)
    elif init_eop_mode == "file":
        init_eop = engine.load_initial_eop(initial_eop_file)
    else:
        raise ValueError("init_eop_mode must be either 'estimate' or 'file'.")

    init_table = engine.eop_to_table(init_eop)

    final_eop, iteration_num, residuals, var_cov, observation_vector = engine.run_least_squares(
        df, init_eop)
    final_table = engine.eop_to_table(final_eop)

    residual_df = pd.DataFrame({"residual_m": residuals})
    observation_df = pd.DataFrame({"observation_minus_computed_m": observation_vector})

    init_table.to_csv(output_dir / "initial_eop.csv", index=False)
    final_table.to_csv(output_dir / "final_eop.csv", index=False)
    residual_df.to_csv(output_dir / "residuals.csv", index=False)
    # observation_df.to_csv(output_dir / "observation_vector.csv", index=False)

    print("Initial EOP:")
    print(init_table.to_string(index=False))

    print("\nIteration count:")
    print(iteration_num)

    print("\nFinal EOP:")
    print(final_table.to_string(index=False))

    print("\nVariance factor:")
    print(var_cov)


if __name__ == "__main__":
    main()