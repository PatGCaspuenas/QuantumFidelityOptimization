from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "scripts" / "data"
OUT_DIR = REPO_ROOT / "figures" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_FILES = {
    50: DATA_DIR / "benchmark_trace_N50_freqspan10_bound100_N50_surrogate_thresh098_40seeds.txt",
    100: DATA_DIR / "benchmark_trace_N100_freqspan10_bound100_N100_surrogate_thresh099_40seeds.txt",
    250: DATA_DIR / "benchmark_trace_N250_freqspan10_bound100_N250_surrogate_thresh0996_40seeds.txt",
}

HALF_FILES = {
    50: DATA_DIR / "benchmark_trace_N50_freqspan10_bound050_N50_surrogate_thresh098_40seeds.txt",
    100: DATA_DIR / "benchmark_trace_N100_freqspan10_bound050_N100_surrogate_thresh099_40seeds.txt",
    250: DATA_DIR / "benchmark_trace_N250_freqspan10_bound050_N250_surrogate_thresh0996_40seeds.txt",
}


def parse_results(path: Path) -> pd.DataFrame:
    header = None
    rows: list[list[str]] = []
    in_results = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped in ("=== All Results ===", "=== INDIVIDUAL RESULTS ==="):
            in_results = True
            header = None
            continue
        if not in_results or not stripped:
            continue
        if line.startswith("Sim\t"):
            header = line.split("\t")
            continue
        if header is not None and line[0].isdigit():
            values = line.split("\t")
            if len(values) >= len(header):
                rows.append(values[: len(header)])
    if header is None:
        raise ValueError(f"Could not parse results from {path}")

    df = pd.DataFrame(rows, columns=header)
    for col in ("Seed", "Iterations", "TotalShots", "Q_det"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_group(files: dict[int, Path], label: str) -> pd.DataFrame:
    frames = []
    for n_shots, path in files.items():
        df = parse_results(path)
        df["N_shots"] = n_shots
        df["scale_label"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def log10_infid(q: pd.Series) -> np.ndarray:
    score = np.clip(q.to_numpy(dtype=float), 0.0, 1.0)
    return np.log10(np.maximum(1.0 - score, 1e-6))


def add_group(ax, df: pd.DataFrame, *, color: str, marker: str) -> None:
    for n_shots in (50, 100, 250):
        sub = df[df["N_shots"] == n_shots].copy()
        x = np.log10(sub["TotalShots"].to_numpy(dtype=float))
        y = log10_infid(sub["Q_det"])

        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            alpha=0.24,
            s=32,
            linewidths=0,
        )
        ax.errorbar(
            float(np.mean(x)),
            float(np.mean(y)),
            xerr=float(np.std(x, ddof=1)),
            yerr=float(np.std(y, ddof=1)),
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=9,
            capsize=4,
            elinewidth=1.8,
            zorder=4,
        )


def print_stats(label: str, df: pd.DataFrame) -> None:
    for n_shots in (50, 100, 250):
        sub = df[df["N_shots"] == n_shots]
        x = np.log10(sub["TotalShots"].to_numpy(dtype=float))
        y = log10_infid(sub["Q_det"])
        print(
            f"{label}\tN={n_shots}\tn={len(sub)}\t"
            f"logshots={np.mean(x):.4f}+/-{np.std(x, ddof=1):.4f}\t"
            f"log10(1-score)={np.mean(y):.4f}+/-{np.std(y, ddof=1):.4f}\t"
            f"median_score={sub['Q_det'].median():.6f}\tmin_score={sub['Q_det'].min():.6f}"
        )


def main() -> None:
    full = read_group(FULL_FILES, "scale_1.0_pm10khz")
    half = read_group(HALF_FILES, "scale_0.5_pm5khz")

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    add_group(ax, full, color="#f6b37f", marker="o")
    add_group(ax, half, color="#c7b4ea", marker="^")

    ax.set_xlabel(r"$\log_{10}(\mathrm{Total\ shots})$", fontsize=15)
    ax.set_ylabel(r"$\log_{10}(1 - \mathrm{Score})$", fontsize=15)
    ax.tick_params(axis="both", direction="in", length=5, labelsize=12)
    ax.grid(False)

    all_x = np.log10(pd.concat([full["TotalShots"], half["TotalShots"]]).to_numpy(dtype=float))
    all_y = np.concatenate([log10_infid(full["Q_det"]), log10_infid(half["Q_det"])])
    ax.set_xlim(float(np.floor(all_x.min() * 10) / 10 - 0.05), float(np.ceil(all_x.max() * 10) / 10 + 0.05))
    ax.set_ylim(float(np.floor(all_y.min() * 10) / 10 - 0.1), float(np.ceil(all_y.max() * 10) / 10 + 0.1))

    print_stats("scale_1.0_pm10khz", full)
    print_stats("scale_0.5_pm5khz", half)

    out_png = OUT_DIR / "scale_full_vs_half_freqspan10_N50_100_250_clean.png"
    out_pdf = OUT_DIR / "scale_full_vs_half_freqspan10_N50_100_250_clean.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
