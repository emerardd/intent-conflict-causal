from __future__ import annotations

from pathlib import Path
from typing import Any


def _svg_text(x: float, y: float, value: str, *, anchor: str = "middle", size: int = 13) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}">{value}</text>'
    )


def _write_fallback_svgs(result: dict[str, Any], output_dir: Path) -> None:
    """Write dependency-free plots when matplotlib is not installed."""
    behavior = result["behavior_rows"]
    groups = [
        ("Authorized", [row["margin"] for row in behavior if row["authorized"]], "#4c78a8"),
        ("Unauthorized", [row["margin"] for row in behavior if not row["authorized"]], "#f58518"),
    ]
    all_values = [value for _, values, _ in groups for value in values]
    low = min(min(all_values), 0.0)
    high = max(max(all_values), 0.0)
    span = high - low or 1.0

    def y_for(value: float) -> float:
        return 350.0 - (value - low) / span * 270.0

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="430" viewBox="0 0 720 430">',
        '<rect width="720" height="430" fill="white"/>',
        _svg_text(360, 28, "Behavioral authorization effect", size=18),
        f'<line x1="90" y1="{y_for(0):.1f}" x2="680" y2="{y_for(0):.1f}" stroke="#555" stroke-dasharray="5 4"/>',
        '<line x1="90" y1="65" x2="90" y2="350" stroke="black"/>',
        '<line x1="90" y1="350" x2="680" y2="350" stroke="black"/>',
    ]
    for tick in range(6):
        value = low + span * tick / 5
        y = y_for(value)
        svg.append(f'<line x1="85" y1="{y:.1f}" x2="90" y2="{y:.1f}" stroke="black"/>')
        svg.append(_svg_text(80, y + 4, f"{value:.1f}", anchor="end", size=11))
    for group_idx, (label, values, color) in enumerate(groups):
        x = 260 + group_idx * 260
        mean = sum(values) / len(values)
        svg.append(f'<line x1="{x - 70}" y1="{y_for(mean):.1f}" x2="{x + 70}" y2="{y_for(mean):.1f}" stroke="{color}" stroke-width="5"/>')
        for idx, value in enumerate(values):
            jitter = ((idx % 13) - 6) * 5.0
            svg.append(f'<circle cx="{x + jitter:.1f}" cy="{y_for(value):.1f}" r="3" fill="{color}" fill-opacity="0.45"/>')
        svg.append(_svg_text(x, 378, label, size=13))
        svg.append(_svg_text(x, y_for(mean) - 10, f"mean {mean:.2f}", size=12))
    svg.append(_svg_text(20, 210, "log p(BLOCK) - log p(EXECUTE)", anchor="middle", size=12).replace('<text ', '<text transform="rotate(-90 20 210)" '))
    svg.append('</svg>')
    (output_dir / "behavior_margin.svg").write_text("\n".join(svg), encoding="utf-8")

    if result.get("representation"):
        rows = result["representation"]["layers"]
        layers = [int(row["layer"]) for row in rows]
        series = [
            ("Validation", [float(row["validation"]["auroc"]) for row in rows], "#4c78a8"),
            ("Held-out test", [float(row["test"]["auroc"]) for row in rows], "#f58518"),
        ]

        def x_layer(layer: int) -> float:
            return 80 + (layer - min(layers)) / (max(layers) - min(layers) or 1) * 570

        def y_auc(value: float) -> float:
            return 350 - value * 270

        svg = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="430" viewBox="0 0 720 430">',
            '<rect width="720" height="430" fill="white"/>',
            _svg_text(360, 28, "Cross-scenario authorization decoding", size=18),
            '<line x1="80" y1="80" x2="80" y2="350" stroke="black"/>',
            '<line x1="80" y1="350" x2="650" y2="350" stroke="black"/>',
            f'<line x1="80" y1="{y_auc(0.5):.1f}" x2="650" y2="{y_auc(0.5):.1f}" stroke="#555" stroke-dasharray="5 4"/>',
        ]
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = y_auc(tick)
            svg.append(_svg_text(70, y + 4, f"{tick:.2f}", anchor="end", size=11))
        for name, values, color in series:
            points = " ".join(f"{x_layer(layer):.1f},{y_auc(value):.1f}" for layer, value in zip(layers, values))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
            for layer, value in zip(layers, values):
                svg.append(f'<circle cx="{x_layer(layer):.1f}" cy="{y_auc(value):.1f}" r="4" fill="{color}"/>')
            legend_x = 450 if name == "Validation" else 560
            svg.append(f'<line x1="{legend_x}" y1="55" x2="{legend_x + 24}" y2="55" stroke="{color}" stroke-width="3"/>')
            svg.append(_svg_text(legend_x + 28, 59, name, anchor="start", size=11))
        for layer in layers:
            svg.append(_svg_text(x_layer(layer), 372, str(layer), size=10))
        svg.append(_svg_text(365, 405, "Decoder layer", size=13))
        svg.append(_svg_text(20, 215, "AUROC", size=13).replace('<text ', '<text transform="rotate(-90 20 215)" '))
        svg.append('</svg>')
        (output_dir / "layer_auroc.svg").write_text("\n".join(svg), encoding="utf-8")

    if result.get("patching"):
        summary = result["patching"]["summary"]
        values = [
            -float(summary["authorized_to_unauthorized_mean_delta"]),
            float(summary["unauthorized_to_authorized_mean_delta"]),
            float(summary["random_sign_aligned_mean"]),
        ]
        labels = ["Auth→Unauth", "Unauth→Auth", "Random control"]
        colors = ["#4c78a8", "#f58518", "#999999"]
        maximum = max(values) * 1.25 or 1.0
        svg = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="430" viewBox="0 0 720 430">',
            '<rect width="720" height="430" fill="white"/>',
            _svg_text(360, 28, "Bidirectional activation-swap effects", size=18),
            '<line x1="90" y1="350" x2="680" y2="350" stroke="black"/>',
            '<line x1="90" y1="70" x2="90" y2="350" stroke="black"/>',
        ]
        for idx, (label, value, color) in enumerate(zip(labels, values, colors)):
            x = 180 + idx * 190
            height = value / maximum * 250
            svg.append(f'<rect x="{x - 45}" y="{350 - height:.1f}" width="90" height="{height:.1f}" fill="{color}"/>')
            svg.append(_svg_text(x, 373, label, size=12))
            svg.append(_svg_text(x, 340 - height, f"{value:.3f}", size=12))
        svg.append(_svg_text(20, 215, "Change toward source decision", size=12).replace('<text ', '<text transform="rotate(-90 20 215)" '))
        svg.append('</svg>')
        (output_dir / "patching_effect.svg").write_text("\n".join(svg), encoding="utf-8")

def write_plots_and_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_status = "not requested"
    try:
        import matplotlib.pyplot as plt

        behavior = result["behavior_rows"]
        authorized = [row["margin"] for row in behavior if row["authorized"]]
        unauthorized = [row["margin"] for row in behavior if not row["authorized"]]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(
            [authorized, unauthorized],
            tick_labels=["Authorized", "Unauthorized"],
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("log p(BLOCK) - log p(EXECUTE)")
        ax.set_title("Behavioral authorization effect")
        fig.tight_layout()
        fig.savefig(output_dir / "behavior_margin.png", dpi=180)
        plt.close(fig)

        if result.get("representation"):
            rows = result["representation"]["layers"]
            layers = [row["layer"] for row in rows]
            val = [row["validation"]["auroc"] for row in rows]
            test = [row["test"]["auroc"] for row in rows]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(layers, val, marker="o", label="Validation")
            ax.plot(layers, test, marker="o", label="Held-out test")
            ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Decoder layer")
            ax.set_ylabel("AUROC")
            ax.set_ylim(0, 1.02)
            ax.legend()
            ax.set_title("Cross-scenario authorization decoding")
            fig.tight_layout()
            fig.savefig(output_dir / "layer_auroc.png", dpi=180)
            plt.close(fig)

        if result.get("patching"):
            summary = result["patching"]["summary"]
            names = ["Authorized→Unauthorized", "Unauthorized→Authorized"]
            values = [
                -summary["authorized_to_unauthorized_mean_delta"],
                summary["unauthorized_to_authorized_mean_delta"],
            ]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(names, values, color=["#4c78a8", "#f58518"])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Sign-aligned change toward source decision")
            ax.set_title("Bidirectional activation-swap effects")
            fig.tight_layout()
            fig.savefig(output_dir / "patching_effect.png", dpi=180)
            plt.close(fig)
        plot_status = "generated with matplotlib"
    except ModuleNotFoundError:
        _write_fallback_svgs(result, output_dir)
        plot_status = "generated as dependency-free SVG files"

    lines = [
        "# Formal experiment report",
        "",
        f"Model: `{result['provenance']['model_name']}`",
        f"Quantization: `{result['provenance']['quantization']}`",
        f"Behavior gate passed: **{result['behavior_gate']['passed']}**",
        f"Plot status: {plot_status}",
        "",
        "## Behavioral positive control",
        "",
    ]
    overall = result["behavior_summary"]["overall"]
    lines.extend(
        [
            f"- Authorized accuracy: {overall['authorized_accuracy']:.3f}",
            f"- Unauthorized accuracy: {overall['unauthorized_accuracy']:.3f}",
            f"- Paired margin effect: {overall['paired_effect_mean']:.4f}",
            f"- Positive scenario fraction: {overall['positive_scenario_fraction']:.3f}",
            "",
        ]
    )
    if result.get("representation"):
        selected = result["representation"]["selected_layer_metrics"]
        permutation = result["representation"]["permutation_test"]
        lines.extend(
            [
                "## Held-out representation",
                "",
                f"- Validation-selected layer: {selected['layer']}",
                f"- Test AUROC: {selected['test']['auroc']:.3f}",
                f"- Train/test direction cosine: {selected['train_test_direction_cosine']:.3f}",
                f"- Cross-dot permutation p-value: {permutation['p_cross_dot_greater_equal']:.4f}",
                "",
            ]
        )
    if result.get("patching"):
        summary = result["patching"]["summary"]
        lines.extend(
            [
                "## Causal activation swap",
                "",
                f"- Authorized→unauthorized raw margin delta: {summary['authorized_to_unauthorized_mean_delta']:.4f}",
                f"- Unauthorized→authorized raw margin delta: {summary['unauthorized_to_authorized_mean_delta']:.4f}",
                f"- Mean sign-aligned causal effect: {summary['sign_aligned_mean']:.4f}",
                f"- 95% scenario-cluster bootstrap CI: [{summary['sign_aligned_ci95'][0]:.4f}, {summary['sign_aligned_ci95'][1]:.4f}]",
                f"- Random-control mean: {summary['random_sign_aligned_mean']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            result["decision"],
            "",
            "The study concerns a functional pre-action authorization representation. It does not establish consciousness or natural deceptive intent.",
        ]
    )
    (output_dir / "formal-report.md").write_text("\n".join(lines), encoding="utf-8")
