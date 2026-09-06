"""Seeded, smooth trajectory fixtures for the intuition notebook."""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type TrajectoryGroups = dict[str, list[FloatArray]]


def smoothstep(u: FloatArray) -> FloatArray:
    """Turn progression values into a smooth ramp from zero to one."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def dart_throws(
    seed: int = 42, n: int = 121
) -> tuple[
    FloatArray, list[FloatArray], list[FloatArray], TrajectoryGroups, dict[str, float | None]
]:
    """Independent train/calibration/test throws and matched counterfactual paths.

    Coordinates depict a planar execution, not a ballistic physics simulation.
    Mild/strong deviations have exact, recorded onset times. Recovery is a
    smooth transient with no terminal displacement.

    Returns time, training throws, calibration throws, test groups, and onsets.
    Time has shape (n,); each throw has shape (n, 2), with x and y columns.
    """
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 1, n)

    def nominal() -> FloatArray:
        """Generate one successful throw as an (n, 2) coordinate array."""
        control_points = np.array([[-5.0, -1.2], [-3.9, 1.4], [-1.7, 1.0], [0.0, 0.0]])
        control_points += rng.normal(0, [0.07, 0.09], control_points.shape)
        # Each row weights the start, two steering points, and endpoint; weights sum to one.
        bezier_weights = np.column_stack(
            ((1 - u) ** 3, 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u**2, u**3)
        )
        noise = np.sin(np.pi * u)[:, None] * (
            np.sin(4 * np.pi * u)[:, None] * rng.normal(0, 0.02, (1, 2))
        )
        # Build each coordinate as a weighted sum of the four control-point coordinates.
        curve = np.zeros((n, 2))
        for index, control_point in enumerate(control_points):
            weights = bezier_weights[:, index]
            curve[:, 0] += weights * control_point[0]  # x coordinates
            curve[:, 1] += weights * control_point[1]  # y coordinates
        return curve + noise

    train = [nominal() for _ in range(14)]
    calibration = [nominal() for _ in range(8)]
    groups = {"Nominal": [], "Recovery": [], "Mild failure": [], "Strong failure": []}
    onsets = {"Nominal": None, "Recovery": 0.20, "Mild failure": 0.66, "Strong failure": 0.30}
    # Each group shares its unperturbed path with the matching nominal index.
    for _ in range(8):
        base = nominal()
        groups["Nominal"].append(base)
        bump = np.sin(np.pi * np.clip((u - 0.20) / 0.65, 0, 1)) ** 2
        groups["Recovery"].append(base + bump[:, None] * [0.05, 0.95])
        late = smoothstep((u - 0.66) / 0.34)
        groups["Mild failure"].append(base + late[:, None] * [0.15, 1.35])
        early = smoothstep((u - 0.30) / 0.70)
        groups["Strong failure"].append(base + early[:, None] * [0.70, 2.65])
    return u, train, calibration, groups, onsets


def sine_trajectories(
    seed: int = 43, n: int = 181
) -> tuple[
    FloatArray,
    list[FloatArray],
    list[FloatArray],
    list[FloatArray],
    TrajectoryGroups,
    dict[str, float],
]:
    """Return time, training, calibration, nominal tests, anomaly groups, and onsets.

    Time has shape (n,); each path has shape (n, 2), with progression x and signal y.
    Anomalies change the same nominal paths only after their recorded onset.
    """
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 1, n)
    x = 6 * np.pi * u

    def nominal() -> FloatArray:
        """Generate one slightly varied sine path as an (n, 2) array."""
        amplitude = rng.normal(1, 0.025)
        phase = rng.normal(0, 0.035)
        frequency = rng.normal(1, 0.004)
        y = amplitude * np.sin(frequency * x + phase)
        y += rng.normal(0, 0.004, n)
        return np.column_stack((x, y))

    train = [nominal() for _ in range(12)]
    calibration = [nominal() for _ in range(8)]
    nominal_test = [nominal() for _ in range(8)]
    onset = 0.40
    ramp = smoothstep((u - onset) / 0.45)
    phase_bump = np.sin(np.pi * np.clip((u - onset) / 0.45, 0, 1)) ** 2
    local_bump = np.sin(np.pi * np.clip((u - 0.55) / 0.22, 0, 1)) ** 4
    groups = {name: [] for name in ["Frequency", "Amplitude", "Phase", "Local bump"]}
    for base in nominal_test:
        # Add a controlled deformation to the same noisy nominal realization.
        phase_extra = 0.9 * 6 * np.pi * np.maximum(u - onset, 0) ** 2
        changes = {
            "Frequency": np.sin(x + phase_extra) - np.sin(x),
            "Amplitude": 0.85 * ramp * np.sin(x),
            "Phase": np.sin(x + 1.0 * phase_bump) - np.sin(x),
            "Local bump": 1.15 * local_bump,
        }
        for name, change in changes.items():
            groups[name].append(base + np.column_stack((np.zeros(n), change)))
    onsets = {name: (0.55 if name == "Local bump" else onset) for name in groups}
    return u, train, calibration, nominal_test, groups, onsets


def sampling_pair(n: int = 361) -> tuple[FloatArray, FloatArray]:
    """Return uniform and dwell-weighted samples of the same curve, each shape (n, 2)."""
    u = np.linspace(0, 1, n)
    grid = np.linspace(0, 1, 10001)
    density = 1 + 18 * np.exp(-(((grid - 0.40) / 0.055) ** 2))
    cumulative = np.cumsum(density)
    cumulative = (cumulative - cumulative[0]) / (cumulative[-1] - cumulative[0])
    dwell_u = np.interp(u, cumulative, grid)

    def path(progression: FloatArray) -> FloatArray:
        """Convert normalized progression values into x and y coordinates."""
        x = 6 * np.pi * progression
        return np.column_stack((x, np.sin(x)))

    return path(u), path(dwell_u)
