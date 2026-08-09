#!/usr/bin/env python3
"""Create dependency-free SVG plots from the Milestone 5 CSV output."""

import argparse
import csv
import math
from pathlib import Path


# Ghia, Ghia & Shin (J. Comput. Phys. 48, 1982), Re = 400,
# vertical centreline u velocity benchmark.
GHIA_Y = [
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
    0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
    0.9531, 0.9609, 0.9688, 0.9766, 1.0000,
]
GHIA_UX = [
    0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
    -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
    0.55892, 0.61756, 0.68439, 0.75837, 1.00000,
]


def read_csv(filename):
    with filename.open(newline="") as file:
        return [
            {name: float(value) for name, value in row.items()}
            for row in csv.DictReader(file)
        ]


def svg_start(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="25" text-anchor="middle" '
        f'font-family="sans-serif" font-size="18">{title}</text>',
    ]


def colour(value, maximum):
    """Map zero to dark blue and maximum speed to yellow."""
    fraction = 0.0 if maximum == 0.0 else min(value / maximum, 1.0)
    red = int(25 + 230 * fraction)
    green = int(45 + 185 * fraction)
    blue = int(120 - 100 * fraction)
    return f"rgb({red},{green},{blue})"


def plot_velocity_vectors(fields, output_directory):
    width = height = 700
    margin = 55
    plot_size = width - 2 * margin
    nx = int(max(row["x"] for row in fields)) + 1
    ny = int(max(row["y"] for row in fields)) + 1
    maximum_speed = max(row["speed"] for row in fields)
    sample_spacing = 6
    arrow_scale = 0.75 * sample_spacing * plot_size / (nx - 1)

    svg = svg_start(width, height, "Lid-driven cavity: velocity vectors")
    svg.append(
        '<defs><marker id="arrow" markerWidth="6" markerHeight="6" '
        'refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" '
        'fill="currentColor"/></marker></defs>'
    )
    svg.append(
        f'<rect x="{margin}" y="{margin}" width="{plot_size}" '
        f'height="{plot_size}" fill="none" stroke="black"/>'
    )

    for row in fields:
        x = int(row["x"])
        y = int(row["y"])
        if x % sample_spacing or y % sample_spacing:
            continue

        screen_x = margin + x * plot_size / (nx - 1)
        screen_y = height - margin - y * plot_size / (ny - 1)
        dx = row["ux"] / maximum_speed * arrow_scale
        dy = -row["uy"] / maximum_speed * arrow_scale
        line_colour = colour(row["speed"], maximum_speed)
        svg.append(
            f'<line x1="{screen_x:.2f}" y1="{screen_y:.2f}" '
            f'x2="{screen_x + dx:.2f}" y2="{screen_y + dy:.2f}" '
            f'stroke="{line_colour}" color="{line_colour}" stroke-width="1.4" '
            'marker-end="url(#arrow)"/>'
        )

    svg.extend(axis_labels(width, height, margin))
    svg.append("</svg>")
    (output_directory / "cavity_velocity_vectors.svg").write_text("\n".join(svg))


def plot_speed(fields, output_directory):
    width = height = 700
    margin = 55
    plot_size = width - 2 * margin
    nx = int(max(row["x"] for row in fields)) + 1
    ny = int(max(row["y"] for row in fields)) + 1
    maximum_speed = max(row["speed"] for row in fields)
    cell_width = plot_size / nx
    cell_height = plot_size / ny

    svg = svg_start(width, height, "Lid-driven cavity: velocity magnitude")
    for row in fields:
        screen_x = margin + row["x"] * cell_width
        screen_y = height - margin - (row["y"] + 1) * cell_height
        svg.append(
            f'<rect x="{screen_x:.2f}" y="{screen_y:.2f}" '
            f'width="{cell_width + 0.1:.2f}" height="{cell_height + 0.1:.2f}" '
            f'fill="{colour(row["speed"], maximum_speed)}"/>'
        )

    svg.extend(axis_labels(width, height, margin))
    svg.append(
        f'<text x="{width - 65}" y="{height - 20}" text-anchor="end" '
        f'font-family="sans-serif" font-size="12">max speed = {maximum_speed:.4f}</text>'
    )
    svg.append("</svg>")
    (output_directory / "cavity_speed.svg").write_text("\n".join(svg))


def axis_labels(width, height, margin):
    return [
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" '
        'font-family="sans-serif">x</text>',
        f'<text x="18" y="{height / 2}" text-anchor="middle" '
        f'transform="rotate(-90 18 {height / 2})" font-family="sans-serif">y</text>',
    ]


def plot_centerline(centerline, output_directory):
    width = 700
    height = 600
    left, right, top, bottom = 80, 670, 45, 545
    x_values = [row["ux_over_lid_velocity"] for row in centerline]
    x_min = min(-0.4, min(x_values))
    x_max = max(1.0, max(x_values))

    def screen_x(value):
        return left + (value - x_min) * (right - left) / (x_max - x_min)

    def screen_y(value):
        return bottom - value * (bottom - top)

    simulation_points = " ".join(
        f'{screen_x(row["ux_over_lid_velocity"]):.2f},'
        f'{screen_y(row["y_over_L"]):.2f}'
        for row in centerline
    )

    svg = svg_start(width, height, "Vertical centreline velocity, Re = 400")
    svg.extend([
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        f'<line x1="{screen_x(0):.2f}" y1="{top}" x2="{screen_x(0):.2f}" '
        'y2="{bottom}" stroke="gray" stroke-dasharray="5 5"/>'.format(bottom=bottom),
        f'<polyline points="{simulation_points}" fill="none" stroke="#1976d2" '
        'stroke-width="3"/>',
        f'<text x="{(left + right) / 2}" y="590" text-anchor="middle" '
        'font-family="sans-serif">ux / lid velocity</text>',
        f'<text x="22" y="{(top + bottom) / 2}" text-anchor="middle" '
        f'transform="rotate(-90 22 {(top + bottom) / 2})" font-family="sans-serif">y / L</text>',
    ])

    # Numerical tick marks make the sign and magnitude readable.
    for tick in [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = screen_x(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" '
                   f'y2="{bottom + 6}" stroke="black"/>')
        svg.append(f'<text x="{x:.2f}" y="{bottom + 23}" text-anchor="middle" '
                   f'font-family="sans-serif" font-size="12">{tick:.1f}</text>')

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = screen_y(tick)
        svg.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" '
                   f'y2="{y:.2f}" stroke="black"/>')
        svg.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                   f'font-family="sans-serif" font-size="12">{tick:.2f}</text>')

    # Benchmark points are deliberately not connected: they are tabulated data.
    for benchmark_x, benchmark_y in zip(GHIA_UX, GHIA_Y):
        svg.append(f'<circle cx="{screen_x(benchmark_x):.2f}" '
                   f'cy="{screen_y(benchmark_y):.2f}" r="4" '
                   'fill="#d32f2f" stroke="white" stroke-width="1"/>')

    svg.extend([
        '<line x1="455" y1="65" x2="490" y2="65" stroke="#1976d2" '
        'stroke-width="3"/><text x="500" y="70" font-family="sans-serif" '
        'font-size="13">LBM simulation</text>',
        '<circle cx="472" cy="88" r="4" fill="#d32f2f"/>'
        '<text x="500" y="93" font-family="sans-serif" font-size="13">'
        'Ghia et al. (1982)</text>',
        '<text x="400" y="525" font-family="sans-serif" font-size="12" '
        'fill="gray">dashed line: ux = 0</text>',
        "</svg>",
    ])
    (output_directory / "centerline_ux.svg").write_text("\n".join(svg))


def interpolate_centerline(centerline, target_y):
    """Linearly interpolate the simulation ux value at a benchmark y/L."""
    for upper in range(1, len(centerline)):
        lower = upper - 1
        y_lower = centerline[lower]["y_over_L"]
        y_upper = centerline[upper]["y_over_L"]
        if target_y <= y_upper:
            fraction = (target_y - y_lower) / (y_upper - y_lower)
            ux_lower = centerline[lower]["ux_over_lid_velocity"]
            ux_upper = centerline[upper]["ux_over_lid_velocity"]
            return ux_lower + fraction * (ux_upper - ux_lower)
    return centerline[-1]["ux_over_lid_velocity"]


def write_benchmark_comparison(centerline, output_directory):
    """Write pointwise Ghia errors and aggregate validation metrics."""
    comparisons = []
    for y, benchmark_ux in zip(GHIA_Y, GHIA_UX):
        simulation_ux = interpolate_centerline(centerline, y)
        error = simulation_ux - benchmark_ux
        comparisons.append((y, benchmark_ux, simulation_ux, error))

    csv_file = output_directory / "benchmark_comparison.csv"
    with csv_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "y_over_L", "ghia_ux_over_lid", "simulation_ux_over_lid",
            "error", "absolute_error",
        ])
        for y, benchmark_ux, simulation_ux, error in comparisons:
            writer.writerow([y, benchmark_ux, simulation_ux, error, abs(error)])

    rmse = math.sqrt(
        sum(error * error for _, _, _, error in comparisons) / len(comparisons)
    )
    maximum_error = max(abs(error) for _, _, _, error in comparisons)

    summary_file = output_directory / "benchmark_summary.txt"
    summary_file.write_text(
        "benchmark=Ghia_Ghia_Shin_1982_Re400\n"
        f"number_of_points={len(comparisons)}\n"
        f"rmse={rmse:.16g}\n"
        f"maximum_absolute_error={maximum_error:.16g}\n"
    )
    print(f"Benchmark RMSE: {rmse:.6f}")
    print(f"Benchmark maximum absolute error: {maximum_error:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Plot Milestone 5 CSV output.")
    parser.add_argument(
        "results_directory",
        nargs="?",
        type=Path,
        default=Path("milestone5_results"),
    )
    args = parser.parse_args()

    fields_file = args.results_directory / "cavity_fields.csv"
    centerline_file = args.results_directory / "centerline_ux.csv"
    if not fields_file.exists() or not centerline_file.exists():
        raise FileNotFoundError("Run the milestone5 executable before plotting.")

    fields = read_csv(fields_file)
    centerline = read_csv(centerline_file)
    plot_velocity_vectors(fields, args.results_directory)
    plot_speed(fields, args.results_directory)
    plot_centerline(centerline, args.results_directory)
    write_benchmark_comparison(centerline, args.results_directory)
    print(f"Plots written to {args.results_directory}")


if __name__ == "__main__":
    main()
